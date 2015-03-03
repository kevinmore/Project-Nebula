/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>



#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuConfig.h>

#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>
#include <Physics2012/Collide/Shape/Query/hkpRayShapeCollectionFilter.h>

#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>
#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCommands.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppLongRayVirtualMachine.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppVirtualMachine.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppObbVirtualMachine.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppSphereVirtualMachine.h>
#include <Physics2012/Internal/Collide/Mopp/Utility/hkpMoppDebugger.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppRayBundleVirtualMachine.h>


#if defined(HK_DEBUG)
static const char HK_MOPP_CODE_ERROR[] = "You can not supply a null MoppCode. Was the MOPP setup correctly?.";
#endif


#if !defined(HK_PLATFORM_SPU)

hkMoppBvTreeShapeBase::hkMoppBvTreeShapeBase( ShapeType type, const hkpMoppCode* code) : hkpBvTreeShape(type, BVTREE_MOPP), m_code(code)
{
	HK_ASSERT2(0xcf45fedb, m_code, HK_MOPP_CODE_ERROR);
	m_code->addReference();

	m_codeInfoCopy = code->m_info.m_offset;
	m_moppData		= code->m_data.begin();
	m_moppDataSize = code->getCodeSize();
}
#endif

#if !defined(HK_PLATFORM_SPU)
void hkMoppBvTreeShapeBase::queryObb( const hkTransform& obbToMopp, const hkVector4& extent, hkReal tolerance, hkArray<hkpShapeKey>& hits ) const
{
	hkpMoppObbVirtualMachine obbMachine;
	HK_ASSERT2(0xcf45fedd, m_code, HK_MOPP_CODE_ERROR);
	obbMachine.queryObb( m_code, obbToMopp, extent, hkFloat32(tolerance), (hkArray<hkpMoppPrimitiveInfo>*)&hits );
}

void hkMoppBvTreeShapeBase::queryAabb( const hkAabb& aabb, hkArray<hkpShapeKey>& hits ) const
{
	hkpMoppCode::CodeInfo& codeInfo = (hkpMoppCode::CodeInfo&)m_codeInfoCopy;

	hkReal width = 16777216.0f/codeInfo.getScale();
	hkVector4 w; w.setAll(width);
	hkVector4 ma; ma.setAdd( codeInfo.m_offset, w);

	hkAabb newAabb;
	newAabb.m_min.setMax(codeInfo.m_offset, aabb.m_min);
	newAabb.m_max.setMin(ma,     aabb.m_max);

	hkpMoppObbVirtualMachine obbMachine;
	HK_ASSERT2(0xcf45fede, m_code, HK_MOPP_CODE_ERROR);
	obbMachine.queryAabb( m_code, newAabb, (hkArray<hkpMoppPrimitiveInfo>*)&hits );
}

#endif

hkUint32 hkMoppBvTreeShapeBase::queryAabbImpl(HKP_SHAPE_VIRTUAL_THIS const hkAabb& aabb, hkpShapeKey* hits, int maxNumKeys ) HKP_SHAPE_VIRTUAL_CONST
{
	const hkMoppBvTreeShapeBase* thisObj = static_cast<const hkMoppBvTreeShapeBase*>(HK_GET_THIS_PTR);

	HK_DECLARE_ALIGNED_LOCAL_PTR( hkpMoppCode, codePtr, 16 );
	hkpMoppCode::CodeInfo& codeInfo = (hkpMoppCode::CodeInfo&)(thisObj->m_codeInfoCopy);
	codePtr->initialize( codeInfo, thisObj->m_moppData, thisObj->m_moppDataSize );

	hkReal width = 16777216.0f/codeInfo.getScale();
	hkVector4 w; w.setAll(width);
	hkVector4 ma; ma.setAdd( codeInfo.m_offset, w);

	hkAabb newAabb;
	newAabb.m_min.setMax(codeInfo.m_offset, aabb.m_min);
	newAabb.m_max.setMin(ma,     aabb.m_max);

	hkpMoppObbVirtualMachine obbMachine;

#if !defined (HK_PLATFORM_SPU)
	hkArray<hkpShapeKey> hitArray( hits, 0 , maxNumKeys);
	obbMachine.queryAabb( codePtr, newAabb, (hkpMoppObbVirtualMachine::hkpPrimitiveOutputArray)&hitArray );
	if (obbMachine.getNumHits() > maxNumKeys)
	{
		HK_WARN(0x67756543, "Hit array is too small and some hkpShapeKeys have been ignored.");
	}
	return obbMachine.getNumHits();
#else
	obbMachine.m_dmaGroup = HK_MOPP_DEFAULT_DMA_GROUP;
	int numHits = obbMachine.queryAabbWithMaxCapacity( codePtr, newAabb, (hkpMoppObbVirtualMachine::hkpPrimitiveOutputArray)hits, maxNumKeys );
	return numHits;
#endif
}

#if !defined(HK_PLATFORM_SPU)

hkpMoppBvTreeShape::hkpMoppBvTreeShape( const hkpShapeCollection* collection, const hkpMoppCode *code ):
		hkMoppBvTreeShapeBase(HKCD_SHAPE_TYPE_FROM_CLASS(hkpMoppBvTreeShape), code ),
		m_child(collection)
{
}

hkpMoppBvTreeShape::~hkpMoppBvTreeShape()
{
}


#endif

#if defined(HK_PLATFORM_SPU)

void hkpMoppBvTreeShape::getChildShapeFromPpu(hkpShapeBuffer& buffer) const
{
	const hkpShape* shapeOnPpu = m_child.getChild();
	int shapeOnPpuSize = m_childSize;

	const hkpShape* shapeOnSpu = (const hkpShape*)g_SpuCollideUntypedCache->getFromMainMemory(shapeOnPpu, shapeOnPpuSize);
	HKP_PATCH_CONST_SHAPE_VTABLE( shapeOnSpu );
	// COPY over to buffer (instead of dmaing to buffer above, since we are returning this data)
	hkString::memCpy16NonEmpty( buffer, shapeOnSpu, ((shapeOnPpuSize+15)>>4) );
}

#endif


void hkpMoppBvTreeShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
#if ! defined (HK_PLATFORM_SPU)
	getShapeCollection()->getAabb( localToWorld, tolerance, out );
#else
	hkpShapeBuffer buffer;
	getShapeCollectionFromPpu(buffer)->getAabb(localToWorld, tolerance, out);
#endif
}


HK_COMPILE_TIME_ASSERT( sizeof( hkpShapeKey ) == sizeof( hkpMoppPrimitiveInfo ) );


hkBool hkpMoppBvTreeShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
{
	HK_TIMER_BEGIN("rcMopp", HK_NULL);
	hkpMoppLongRayVirtualMachine longray;

	HK_DECLARE_ALIGNED_LOCAL_PTR( hkpMoppCode, codePtr, HK_REAL_ALIGNMENT );
	hkpMoppCode::CodeInfo& codeInfo = (hkpMoppCode::CodeInfo&)m_codeInfoCopy;
	codePtr->initialize(codeInfo, m_moppData, m_moppDataSize );

#if ! defined (HK_PLATFORM_SPU)
	hkBool result = longray.queryLongRay( getShapeCollection(), codePtr, input, results);
#else
	hkpShapeBuffer buffer;
	hkBool result = longray.queryLongRay( getShapeCollectionFromPpu(buffer), codePtr, input, results);
#endif
	HK_TIMER_END();
	return result;
}


hkVector4Comparison hkpMoppBvTreeShape::castRayBundle(const hkpShapeRayBundleCastInput& input, hkpShapeRayBundleCastOutput& results, hkVector4ComparisonParameter mask) const
{
	HK_TIMER_BEGIN("rcBundleMopp", HK_NULL);
	hkpMoppRayBundleVirtualMachine rayBundle;

	HK_DECLARE_ALIGNED_LOCAL_PTR( hkpMoppCode, codePtr, 16 );
	hkpMoppCode::CodeInfo& codeInfo = (hkpMoppCode::CodeInfo&)m_codeInfoCopy;
	codePtr->initialize(codeInfo, m_moppData, m_moppDataSize );

	
#if ! defined (HK_PLATFORM_SPU)
	hkVector4Comparison hits = rayBundle.queryRayBundle( getShapeCollection(), codePtr, input, results, mask);
#else
	hkpShapeBuffer buffer;
	hkVector4Comparison hits = rayBundle.queryRayBundle( getShapeCollectionFromPpu(buffer), codePtr, input, results, mask);
#endif
	HK_TIMER_END();

	return hits;
}

void hkpMoppBvTreeShape::castRayWithCollector(const hkpShapeRayCastInput& input, const hkpCdBody& cdBody, hkpRayHitCollector& collector) const
{
	HK_TIMER_BEGIN("rcMopp", HK_NULL);
	hkpMoppLongRayVirtualMachine longray;

	HK_DECLARE_ALIGNED_LOCAL_PTR( hkpMoppCode, codePtr, 16 );
	hkpMoppCode::CodeInfo& codeInfo = (hkpMoppCode::CodeInfo&)m_codeInfoCopy;
	codePtr->initialize(codeInfo, m_moppData, m_moppDataSize );

#if ! defined (HK_PLATFORM_SPU)
	longray.queryLongRay( getShapeCollection(), codePtr, input, cdBody, collector );
#else
	hkpShapeBuffer buffer;
	const hkpShapeCollection* collection = getShapeCollectionFromPpu(buffer);
	longray.queryLongRay( collection, codePtr, input, cdBody, collector );
#endif
	HK_TIMER_END();
}

#if !defined(HK_PLATFORM_SPU)

int hkpMoppBvTreeShape::calcSizeForSpu(const CalcSizeForSpuInput& input, int spuBufferSizeLeft) const
{
	if ( ( m_code->m_buildType != hkpMoppCode::BUILT_WITH_CHUNK_SUBDIVISION) && ( m_code->m_buildType != hkpMoppCode::BUILT_WITHOUT_CHUNK_SUBDIVISION ) )
	{
		HK_WARN( 0x54e4345d, "MOPP code build type not correctly. Forcing this shape onto PPU" );
		return -1;
	}

	// SPU can't handle anything but subdivided shapes
	if ( m_code->m_buildType == hkpMoppCode::BUILT_WITHOUT_CHUNK_SUBDIVISION )
	{
		HK_WARN( 0xdafe5c0d, "Can't put this shape onto the SPU. You should build the MOPP with hkpMoppCompilerInput::m_enableChunkSubdivision = true." );
		return -1;
	}

	int childSize = m_child.getChild()->calcSizeForSpu(input, HK_SPU_AGENT_SECTOR_JOB_MAX_SHAPE_SIZE);
	if ( childSize < 0 )
	{
		// Child shape will print a more detailed error message (with a reason).
		HK_WARN(0xdbc05911, "hkpMoppBvTreeShape child cannot be processed on SPU.");
		return -1;
	}

	if ( childSize > HK_SPU_AGENT_SECTOR_JOB_MAX_SHAPE_SIZE )
	{
		// early out if cascade will not fit into spu's shape buffer
		HK_WARN(0xdbc05912, "hkpMoppBvTreeShape child will not fit on SPU.");
		return -1;
	}

	// the spu will need this value to properly dma the child shape in one go
	m_childSize = childSize;

	// if child is not consecutive in memory, restart size calculation with just us
	return sizeof(*this);
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
