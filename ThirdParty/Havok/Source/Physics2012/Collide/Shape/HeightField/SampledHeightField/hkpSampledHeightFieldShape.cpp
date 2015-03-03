/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Shape/HeightField/SampledHeightField/hkpSampledHeightFieldShape.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

#include <Physics2012/Collide/Shape/Query/hkpRayHitCollector.h>

#include <Physics2012/Collide/Shape/HeightField/StorageSampledHeightField/hkpStorageSampledHeightFieldShape.h>
#include <Physics2012/Collide/Shape/HeightField/CompressedSampledHeightField/hkpCompressedSampledHeightFieldShape.h>

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuConfig.h>

#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastTriangle.h>
#include <Geometry/Internal/Algorithms/Intersect/hkcdIntersectRayAabb.h>

#ifdef HK_PLATFORM_SPU
hkpSampledHeightFieldShape::HeightFieldFuncs hkpSampledHeightFieldShape::s_heightFieldFunctions[hkpSampledHeightFieldShape::HEIGHTFIELD_MAX_ID];
#endif

// Default value for the ray and sphere casting function pointers.
hkpSampledHeightFieldShape::RayCastInternalFunc hkpSampledHeightFieldShape::s_rayCastFunc = HK_NULL;
#ifndef HK_PLATFORM_SPU
hkpSampledHeightFieldShape::SphereCastInternalFunc hkpSampledHeightFieldShape::s_sphereCastFunc = HK_NULL;
#endif


#if !defined(HK_PLATFORM_SPU)

hkpSampledHeightFieldShape::hkpSampledHeightFieldShape( const hkpSampledHeightFieldBaseCinfo& ci, HeightFieldType hfType )
: hkpHeightFieldShape(HKCD_SHAPE_TYPE_FROM_CLASS(hkpSampledHeightFieldShape)), m_heightfieldType(hfType)
{
	m_xRes = ci.m_xRes;
	m_zRes = ci.m_zRes;

	m_useProjectionBasedHeight = ci.m_useProjectionBasedHeight;

		// setting the extents based on the current heightfield dimensions
	m_extents.set( ci.m_xRes-1.0f, ci.m_maxHeight - ci.m_minHeight, ci.m_zRes-1.0f );
	m_extents.mul( ci.m_scale );

	if ( ci.m_minHeight > ci.m_maxHeight )
	{
		m_raycastMinY = -HK_REAL_MAX; // We can't assume anything about the heights
		m_raycastMaxY = HK_REAL_MAX;
		m_heightCenter = -1.0f;
		m_extents(1) = -1.0f;
	}
	else
	{
		m_raycastMinY = ci.m_minHeight;
		m_raycastMaxY = ci.m_maxHeight;
		m_heightCenter = 0.5f * ( ci.m_minHeight + ci.m_maxHeight ) * ci.m_scale(1); 
	}
	
	m_intToFloatScale = ci.m_scale;

	hkVector4 v; v.setXYZ_W( ci.m_scale, hkSimdReal_1 );	// avoid div-by-zero exception in w
	m_floatToIntScale.setReciprocal<HK_ACC_FULL,HK_DIV_SET_MAX>( v );

	//
	//	Correct for incorrect float to int conversion
	//
	{
		hkReal c; hkVector4Util::getFloatToInt16FloorCorrection(c);
		hkVector4 c4; 
		c4.setMul( hkSimdReal::fromFloat(c), m_intToFloatScale );
		c4.zeroComponent<1>();
		m_floatToIntOffsetFloorCorrected = c4;
	}
	HK_ASSERT2(0x746bd538,  m_xRes < 0x8000 && m_zRes < 0x8000, "The current heightfield dimension is limited to 16k * 16k cells" );
	m_coarseness = 0;
}

hkpSampledHeightFieldShape::hkpSampledHeightFieldShape( hkFinishLoadedObjectFlag flag )
:	hkpHeightFieldShape(flag), m_coarseTreeData(flag)
{
	if (flag.m_finishing)
	{
		setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpSampledHeightFieldShape));
	}
}

hkpSampledHeightFieldShape::~hkpSampledHeightFieldShape()
{}

void hkpSampledHeightFieldShape::collideSpheres( const CollideSpheresInput& input, SphereCollisionOutput* outputArray) const
{
	HK_WARN_ALWAYS(0x4b50036a, "You must implement collideSpheres!");
}

hkReal hkpSampledHeightFieldShape::getHeightAtImpl( int x, int z ) const
{
	HK_WARN_ALWAYS(0x4b50036b, "You must implement getHeightAtImpl!");
	return 0.0f;
}

hkBool hkpSampledHeightFieldShape::getTriangleFlipImpl() const
{
	HK_WARN_ALWAYS(0x4b50036c, "You must implement getTriangleFlipImpl!");
	return false;
}

#endif

void hkpSampledHeightFieldShape::getCinfo( hkpSampledHeightFieldBaseCinfo& cinfo ) const
{
	HK_ASSERT2(0x4b50036d, false, "getCinfo for Sampled Height Field is not implemented yet.");
}

void hkpSampledHeightFieldShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	// Only re-calculate the AABB of the height-field on the PPU. We are assuming here
	// that a) the AABB is calculated when the height-field is added to the world and 
	// b) that the AABB is updated if any height values change.
#ifndef HK_PLATFORM_SPU
	hkReal minY = getHeightAt(0,0);
	hkReal maxY = minY;
	if ( m_extents(1) < 0 )
	{
		for ( int i = 0; i < m_xRes; i++ )
		{
			for (int j = 0; j < m_zRes; j++ )
			{
				hkReal z = getHeightAt(i,j);
				minY = hkMath::min2( minY, z );
				maxY = hkMath::max2( maxY, z );
			}
		}

		hkpSampledHeightFieldShape* base = const_cast<hkpSampledHeightFieldShape*>(this);
		base->m_raycastMinY = minY;
		base->m_raycastMaxY = maxY;

		minY *= this->m_intToFloatScale(1);
		maxY *= this->m_intToFloatScale(1);

		// maybe the z signRayDir is reverse, so correct this
		hkReal h = minY;
		minY = hkMath::min2( minY, maxY );	
		maxY = hkMath::max2( h,    maxY );

		base->m_heightCenter = 0.5f * ( minY + maxY );
		base->m_extents(1) = maxY - minY;
	}
#else
	HK_ASSERT2( 0x264b4fd2, m_extents(1) > 0.0f, "AABB of heightfield must be calculated before simulating on the SPU." );
#endif


	hkVector4 center; center.setMul( hkSimdReal_Half, m_extents );
	hkVector4 halfExtents = center;

	center(1) = m_heightCenter;

	hkAabbUtil::calcAabb( localToWorld, halfExtents, center, hkSimdReal::fromFloat(tolerance), out );
}

#ifdef HK_PLATFORM_SPU

void HK_CALL hkpSampledHeightFieldShape::registerHeightFieldFunctions()
{
	s_heightFieldFunctions[HEIGHTFIELD_STORAGE].m_getHeightAtFunc = static_cast<GetHeightAtFunc> (&hkpStorageSampledHeightFieldShape::getHeightAtImpl);
	s_heightFieldFunctions[HEIGHTFIELD_STORAGE].m_getTriangleFlipFunc = static_cast<GetTriangleFlipFunc> (&hkpStorageSampledHeightFieldShape::getTriangleFlipImpl);

	s_heightFieldFunctions[HEIGHTFIELD_COMPRESSED].m_getHeightAtFunc = static_cast<GetHeightAtFunc> (&hkpCompressedSampledHeightFieldShape::getHeightAtImpl);
	s_heightFieldFunctions[HEIGHTFIELD_COMPRESSED].m_getTriangleFlipFunc = static_cast<GetTriangleFlipFunc> (&hkpCompressedSampledHeightFieldShape::getTriangleFlipImpl);

	// Users may register their custom implementation here.
	// s_heightFieldFunctions[HEIGHTFIELD_USER].m_getHeightAtFunc = ...
}

#endif



void hkpSampledHeightFieldShape::getHeightAndNormalAt( int xPos, int zPos, hkReal subX, hkReal subZ, hkVector4& normalOut, hkReal& heightOut, int& triangleIndexOut ) const
{
	_getHeightAndNormalAt( xPos, zPos, subX, subZ, normalOut,heightOut, triangleIndexOut );
}


void hkpSampledHeightFieldShape::getCoarseMinMax(int level, int x, int z, hkVector4& minOut, hkVector4& maxOut) const
{
	HK_ASSERT(0x42893575, !m_coarseTreeData.isEmpty());
	HK_ASSERT(0x42893572, level>=m_coarseness);
	const CoarseMinMaxLevel& coarseLevel = m_coarseTreeData[level-m_coarseness];
	if (x>=coarseLevel.m_xRes || z>=coarseLevel.m_zRes) return;	// Can happen for non power-of-two height fields.
	int index = 2*(x*coarseLevel.m_zRes+z);
	minOut = coarseLevel.m_minMaxData[index];
	maxOut = coarseLevel.m_minMaxData[index+1];
}




namespace
{
	struct NearestHitCollector : public hkpRayHitCollector
	{
		NearestHitCollector(hkpShapeRayCastOutput& output) : m_hasHit(false), m_output(output)
		{
			m_earlyOutHitFraction = output.m_hitFraction;
		}

		virtual void addRayHit( const hkpCdBody& cdBody, const hkpShapeRayCastCollectorOutput& hitInfoIn )
		{
			if ( hitInfoIn.m_hitFraction < m_earlyOutHitFraction )
			{
				const hkpShapeRayCastOutput& hitInfo = static_cast<const hkpShapeRayCastOutput&>(hitInfoIn);
				m_earlyOutHitFraction = hitInfo.m_hitFraction;
				m_output.m_normal = hitInfo.m_normal;
				m_output.m_hitFraction = hitInfo.m_hitFraction;
				m_output.m_extraInfo = hitInfoIn.m_extraInfo;
				m_output.setKey( HK_INVALID_SHAPE_KEY );
				m_hasHit = true;
			}
		}

		hkBool m_hasHit;
		hkpShapeRayCastOutput& m_output;
	};
}

namespace
{
	struct RotateNormalHitCollector : public hkpRayHitCollector
	{
		RotateNormalHitCollector(hkpRayHitCollector& collector) : m_collector(collector)
		{
			m_earlyOutHitFraction = collector.m_earlyOutHitFraction;
		}

		virtual void addRayHit( const hkpCdBody& cdBody, const hkpShapeRayCastCollectorOutput& hitIn )
		{
			hkpShapeRayCastCollectorOutput hitOut = hitIn;
			hitOut.m_normal._setRotatedDir( cdBody.getTransform().getRotation(), hitIn.m_normal );
			m_collector.addRayHit( cdBody, hitOut );
			m_earlyOutHitFraction = m_collector.m_earlyOutHitFraction;
		}
		hkpRayHitCollector& m_collector;
	};
}

hkBool hkpSampledHeightFieldShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results ) const
{
	NearestHitCollector myCollector(results);
	hkpCdBody* cdBody = HK_NULL;
	HK_ASSERT2(0x7505444d, s_rayCastFunc != HK_NULL, "No ray cast function registered!");
	(this->*s_rayCastFunc)( input, *cdBody, myCollector );
	return myCollector.m_hasHit;
}

void hkpSampledHeightFieldShape::castRayWithCollector(const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector) const
{
	RotateNormalHitCollector myCollector(collector);
	HK_ASSERT2(0x7505444e, s_rayCastFunc != HK_NULL, "No ray cast function registered!");
	(this->*s_rayCastFunc)( input, cdBody, myCollector);
}

// Sphere cast is not available on SPU
#ifndef HK_PLATFORM_SPU
void hkpSampledHeightFieldShape::castSphere( const hkpSphereCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector ) const
{
	// We approximate the sphere cast with a lowered raycast
	hkVector4 delta; delta.set(0.0f, -input.m_radius, 0.0f);
	delta.setFlipSign(delta, m_intToFloatScale.getComponent<1>());	// negate delta if m_intToFloatScale.y is negative

	hkpSphereCastInput newInput = input;
	newInput.m_from.setAdd(input.m_from, delta);
	newInput.m_to.setAdd(input.m_to, delta);

	HK_ASSERT2(0x7505444f, s_sphereCastFunc != HK_NULL, "No sphere cast function registered!");
	(this->*s_sphereCastFunc)(newInput, cdBody, collector );
}
#endif

/*
 * Havok SDK - Product file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
