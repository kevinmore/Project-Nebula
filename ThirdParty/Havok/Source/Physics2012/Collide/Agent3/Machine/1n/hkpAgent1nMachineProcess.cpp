/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>

#if defined(HK_PLATFORM_SPU)
#	include <Common/Base/Memory/PlatformUtils/Spu/SpuDmaCache/hkSpu4WayCache.h>
#	include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#	include <Common/Base/Container/ArraySpu/hkArraySpu.h>
#endif

#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>

#include <Physics2012/Internal/Collide/Gjk/hkpGsk.h>
#include <Physics2012/Dynamics/Collide/Deprecated/Dispatch/hkpCollideCallbackDispatcher.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuConfig.h>

#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>
#include <Physics2012/Collide/Dispatch/Agent3Bridge/hkpAgent3Bridge.h>

#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>

#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nTrack.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachine.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>

#include <Physics2012/Collide/Shape/hkpShape.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpCpuSingleContainerIterator.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpCpuDoubleContainerIterator.h>

#if defined HK_PLATFORM_PS3_SPU
#	include <Common/Base/Monitor/Spu/hkSpuMonitorCache.h>
#endif


		// for spu alloc/free
#define DEALLOCATE_SECTOR(s) hkDeallocateChunk<hkpAgent1nSector>(s,1,HK_MEMORY_CLASS_DYNAMICS)

#define ON_1N_MACHINE(code) code
#define ON_NM_MACHINE(code)

#define EXTRACT_CHILD_SHAPES(containerIterator, bodyA, bodyB) containerIterator.setShape( bodyB );

#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
// When data is copied from the extension buffer into a sector, pointers into the buffer belonging to the potential contact info (used for deprecated welding)
// need to be fixed up.
static HK_FORCE_INLINE void fixupPotentialContactPointers( const hkUchar* extensionBuffer, void* dst, int bytesUsedInExtensionBuffer, hkpProcessCollisionOutput::PotentialInfo* potentialInfo )
{
	if ( potentialInfo )
	{
		// We may need to fix up the potential contacts.
		for ( hkpProcessCollisionOutput::ContactRef* contactRef = &potentialInfo->m_potentialContacts[0]; contactRef < potentialInfo->m_firstFreePotentialContact; ++contactRef )
		{
			hkLong offsetInExtensionBuffer = hkGetByteOffset( &extensionBuffer[0], contactRef->m_agentEntry );
			// Check that the pointer points into the extension buffer.
			if ( ( 0 <= offsetInExtensionBuffer ) && ( offsetInExtensionBuffer < bytesUsedInExtensionBuffer ) )
			{
				contactRef->m_agentEntry = (hkpAgentEntry*) hkAddByteOffset( dst, offsetInExtensionBuffer );
				contactRef->m_agentData = (hkpAgentData*) hkAddByteOffset( dst, hkGetByteOffset( &extensionBuffer[0], contactRef->m_agentData ) );
			}
		}
	}
}
#endif // defined(HK_1N_MACHINE_SUPPORTS_WELDING)

//
// Include 1N version
//
#define HK_PROCESS_FUNC_NAME(X) X
#define HK_CONTAINER_ITERATOR_TYPE hkpCpuSingleContainerIterator
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachineProcess.cxx>


//#undef HK_PROCESS_FUNC_NAME
//#undef HK_CONTAINER_ITERATOR_TYPE
// #define HK_1N_PROCESS_FUNC(X) X##_Compound
// #define HK_1N_ITERATOR_TYPE hkpCpuSingleContainerIterator
// #include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachineProcess.cxx>
// #undef HK_PROCESS_FUNC_NAME
// #undef HK_CONTAINER_ITERATOR_TYPE


#undef  ON_1N_MACHINE
#define ON_1N_MACHINE(code)
#undef  ON_NM_MACHINE
#define ON_NM_MACHINE(code) code

#if defined HK_1N_MACHINE_SUPPORTS_WELDING
#	undef HK_1N_MACHINE_SUPPORTS_WELDING
#endif

#undef  EXTRACT_CHILD_SHAPES
#define EXTRACT_CHILD_SHAPES(containerIterator, bodyA, bodyB) containerIterator.setShapes( bodyA, bodyB );

HK_COMPILE_TIME_ASSERT( sizeof(hkpAgent1nSector) == 512);

//
// Include NM version
//
#undef HK_PROCESS_FUNC_NAME
#undef HK_CONTAINER_ITERATOR_TYPE
#define HK_PROCESS_FUNC_NAME(X) X
#define HK_CONTAINER_ITERATOR_TYPE hkpCpuDoubleContainerIterator
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachineProcess.cxx>


// #undef HK_PROCESS_FUNC_NAME
// #undef HK_CONTAINER_ITERATOR_TYPE
// #define HK_PROCESS_FUNC_NAME(X) X##_Compound
// #define HK_CONTAINER_ITERATOR_TYPE hkpCpuDoubleContainerIterator
// #include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachineProcess.cxx>

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
