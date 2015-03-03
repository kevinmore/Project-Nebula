/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>

#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastBox.h>

#if defined HK_COMPILER_MSVC
	// C4701: local variable 'lastNormal' and 'hitNormal' may be used without having been initialized
#	pragma warning(disable: 4701)
#endif

static inline hkBool32 HK_CALL isPositive(const hkVector4& v ){	return v.greaterZero().allAreSet<hkVector4ComparisonMask::MASK_XYZ>(); }

#if !defined(HK_PLATFORM_SPU)

#if (HK_POINTER_SIZE==4) && (HK_NATIVE_ALIGNMENT==16) && !defined(HK_REAL_IS_DOUBLE) && !defined(HK_COMPILER_HAS_INTRINSICS_NEON)
HK_COMPILE_TIME_ASSERT( sizeof(hkpBoxShape) == 48 );
#endif

hkpBoxShape::hkpBoxShape( const hkVector4& halfExtents, hkReal radius )
:	hkpConvexShape(HKCD_SHAPE_TYPE_FROM_CLASS(hkpBoxShape), radius), m_halfExtents(halfExtents)
{
	const hkSimdReal minExtent = m_halfExtents.horizontalMin<3>();
	m_halfExtents.setW(minExtent);

	HK_ASSERT2(0x1cda850c,  isPositive(m_halfExtents), "hkpBoxShape passed a NONPOSITIVE-valued extent");
}

//
//	Serialization constructor

hkpBoxShape::hkpBoxShape( hkFinishLoadedObjectFlag flag )
:	hkpConvexShape(flag)
{
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpBoxShape));
}

hkpBoxShape::~hkpBoxShape()
{
}

	// hkpConvexShape interface implementation.
void hkpBoxShape::getFirstVertex(hkVector4& v) const
{
	v = m_halfExtents;
}

#endif // !defined(HK_PLATFORM_SPU)

void hkpBoxShape::setHalfExtents(const hkVector4& halfExtents)
{
	HK_ASSERT2(0x5e756678,  isPositive(halfExtents), "hkpBoxShape passed a NONPOSITIVE-valued extent");
	m_halfExtents.setXYZ_W(halfExtents, halfExtents.horizontalMin<3>());
}

void hkpBoxShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	hkAabbUtil::calcAabb(localToWorld, m_halfExtents, hkSimdReal::fromFloat(tolerance + m_radius), out);
}


void hkpBoxShape::getSupportingVertex(hkVector4Parameter direction, hkcdVertex& supportingVertexOut) const
{
	hkVector4 support;
	support.setFlipSign(m_halfExtents, direction);

	// get a unique number that specifies which corner we've got.
	int vertexID = support.lessZero().getMask<hkVector4ComparisonMask::MASK_XYZ>();
	support.setInt24W( vertexID );
	static_cast<hkVector4&>(supportingVertexOut) = support;

	HK_SPU_UPDATE_STACK_SIZE_TRACE();
}

void hkpBoxShape::getCentre(hkVector4& centreOut) const
{
	centreOut.setZero();
}



HK_COMPILE_TIME_ASSERT( hkVector4ComparisonMask::MASK_XYZ == 0x07 );

void hkpBoxShape::convertVertexIdsToVertices(const hkpVertexId* ids, int numIds, hkcdVertex* verticesOut) const
{
	for (int i = numIds-1; i>=0; i--)
	{
		int bits = ids[0];
		HK_ASSERT2(0x347791, (bits & hkVector4ComparisonMask::MASK_XYZ) == ids[0], "illegal vertex id");
		hkVector4Comparison mask; mask.set((hkVector4ComparisonMask::Mask)bits);
		verticesOut[0].setFlipSign(m_halfExtents, mask);
		verticesOut[0].setInt24W( bits );
		verticesOut++;
		ids++;
	}
}

const hkSphere* hkpBoxShape::getCollisionSpheres(hkSphere* sphereBuffer) const
{
	hkSphere* s = sphereBuffer;

	hkSimdReal r; r.load<1>(&m_radius);
	hkVector4 radius; radius.setXYZ_W(m_halfExtents, r);

	for (int i=0; i<8; ++i)
	{
		hkVector4Comparison mask; 
		mask.set((hkVector4ComparisonMask::Mask)i);
		hkVector4 v; 
		v.setFlipSign(radius, mask);
		s[i].setPositionAndRadius(v);
	}

	return sphereBuffer;
}




// Boundary coordinate sign bit meanings for the "AND" of the 'outcodes'
// sign (b)	sign (t)
// 0		0		whole segment inside box
// 1		0		b is outside box, t is in
// 0		1		b is inside box, t is out
// 1		1		whole segment outside box

// ray-box intersection with ideas from 'A trip down the graphics pipeline', Jim Blinn
// if ray starts within the box, no intersection is returned
// return 1 for success, 0 for no intersection. when success then hitpoint and hitnormal is filled



hkBool hkpBoxShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
{
	HK_TIMER_BEGIN("rcBox", HK_NULL);

	hkSimdReal radiusExtents; radiusExtents.load<1>(&m_radius);
	hkVector4 positiveHalfExtents; positiveHalfExtents.setAdd( m_halfExtents, radiusExtents );

	hkcdRay ray; ray.setEndPoints( input.m_from, input.m_to );

#if defined(HK_PLATFORM_SPU)
	hkSimdReal fraction = hkSimdReal::fromFloat(results.m_hitFraction); // fix for hkPadSpu
#else
	hkSimdReal fraction; fraction.load<1>(&results.m_hitFraction);
#endif

	hkVector4 normal;
	hkBool32 hit = hkcdRayCastBox(ray, positiveHalfExtents, &fraction, &normal);

	// Debug!
// 	hkSimdReal fractionOld = fraction;
// 	hkVector4 normalOld;
// 	hkBool32 hitOld = hkcdSegmentBoxIntersectDeprecated(ray, positiveHalfExtents, fractionOld, normalOld);
// 	hkBool diff = (hitOld ? true : false) != (hit ? true : false);
// 	if ( diff )
// 	{
// 		diff = diff;
// 	}
	
	if ( hit )
	{
		results.m_normal = normal;
#if defined(HK_PLATFORM_SPU)
		results.m_hitFraction = fraction.getReal(); // fix for hkPadSpu
#else
		fraction.store<1>(&results.m_hitFraction);
#endif
		results.setKey( HK_INVALID_SHAPE_KEY );

		HK_TIMER_END();
		return true;
	}

	HK_TIMER_END();
	return false;
}

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
