/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COLLIDE_SHARED_DATA_H
#define HKNP_COLLIDE_SHARED_DATA_H

#include <Physics/Physics/hknpConfig.h>
#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.h>
#include <Physics/Physics/Collide/Filter/hknpCollisionFilter.h>
#include <Physics/Physics/Collide/Shape/TagCodec/hknpShapeTagCodec.h>

class hkBlockStreamAllocator;
struct hknpSolverInfo;
class hknpSimulationContext;
class hkIntSpaceUtil;
class hknpCdCacheStream;


/// List of parameters which is used by the modifiers
class hknpModifierSharedData
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpModifierSharedData );

	public:

		hkPadSpu<hknpSpaceSplitter*>		m_spaceSplitter;
		hkPadSpu<hkUint32>					m_spaceSplitterSize;
		hkPadSpu<const hknpSolverInfo*>		m_solverInfo;
		hkSimdReal							m_collisionTolerance;
};


/// List of parameters which is used by the collide pipeline
class hknpInternalCollideSharedData : public hknpModifierSharedData
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpInternalCollideSharedData );

		hknpInternalCollideSharedData() {}

#if !defined(HK_PLATFORM_SPU)
		/// Constructor used for single threaded simulation
		hknpInternalCollideSharedData( hknpWorld* world );
#endif

	public:

		hkPadSpu<hknpBody*>				m_bodies;
		hkPadSpu<hknpMotion*>			m_motions;

		hkPadSpu<hkIntSpaceUtil*>		m_intSpaceUtil;

		hkPadSpu<hknpCdCacheStream*>	m_cdCacheStreamInOnPpu;
		hkPadSpu<hknpCdCacheStream*>	m_cdCacheStreamIn2OnPpu;
		hkPadSpu<hknpCdCacheStream*>	m_childCdCacheStreamInOnPpu;
		hkPadSpu<hknpCdCacheStream*>	m_childCdCacheCacheStreamIn2OnPpu;

		hkPadSpu<hkUint32>				m_enableRebuildCdCaches1;	///< Set to 0 to ignore hknpBody::FLAG_REBUILD_CD_CACHES for input stream 1, ~0 otherwise
		hkPadSpu<hkUint32>				m_enableRebuildCdCaches2;	///< Set to 0 to ignore hknpBody::FLAG_REBUILD_CD_CACHES for input stream 2, ~0 otherwise
};


/// List of parameters which is used by the collide pipeline.
class hknpCollideSharedData : public hknpInternalCollideSharedData
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCollideSharedData );

	public:

		hkPadSpu<const hknpBodyQuality*>	m_qualities;
		hkPadSpu<hkUint32>					m_numQualities;
		hkPadSpu<const hknpMaterial*>		m_materials;
		hkPadSpu<hkUint32>					m_numMaterials;
		hkPadSpu<hkBlockStreamAllocator*>	m_heapAllocator;
		hkPadSpu<hkBlockStreamAllocator*>	m_tempAllocator;
		hkPadSpu<hknpSimulationContext*>	m_simulationContext;	// only for PPU
#if defined(HK_PLATFORM_HAS_SPU)
		hkPadSpu<hknpModifierManager*>						m_modifierManager;
		hkPadSpu<const hknpCollisionFilter*>				m_collisionFilter;
		hkPadSpu<const hknpShapeTagCodec*>					m_shapeTagCodec;
		hkPadSpu<hknpModifierFlags>							m_globalModifierFlags;
		hkEnum<hknpCollisionFilter::FilterType, hkUint8>	m_collisionFilterType;
		hkEnum<hknpShapeTagCodec::CodecType, hkUint8>		m_shapeTagCodecType;
#endif
};


#endif // HKNP_COLLIDE_SHARED_DATA_H

/*
 * Havok SDK - Base file, BUILD(#20130912)
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
