/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Internal/hkpInternal.h>
#include <Physics2012/Internal/Dynamics/World/Simulation/Continuous/ToiResourceMgr/Default/hkpDefaultToiResourceMgr.h>


#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>

#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>

#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkTypeInfo.h>

HK_REFLECTION_DEFINE_STUB_VIRTUAL(hkpDefaultToiResourceMgr, hkReferencedObject);


#define HK_TOI_SCRATCH_MEMORY_SIZE (1024*128)

hkpDefaultToiResourceMgr::hkpDefaultToiResourceMgr()
{
	m_defaultScratchpadSize = HK_TOI_SCRATCH_MEMORY_SIZE;
}

int hkpDefaultToiResourceMgr::getScratchpadCapacity()
{
	return m_defaultScratchpadSize;
}

HK_COMPILE_TIME_ASSERT( hkpConstraintInstance::NUM_PRIORITIES == 6 );
HK_COMPILE_TIME_ASSERT( hkpConstraintSolverResources::NUM_PRIORITY_CLASSES == 3 );

const hkUint8 hkpDefaultToiResourceMgr::s_priorityClassMap[hkpConstraintInstance::NUM_PRIORITIES] = { 0, 0, 0, 0, 1, 2 };
const hkReal hkpDefaultToiResourceMgr::s_priorityClassRatios[hkpConstraintSolverResources::NUM_PRIORITY_CLASSES] = { 0.3333f, 0.3333f, 0.3333f };

// 7.1 behavior.
//const hkUint8 hkpDefaultToiResourceMgr::s_priorityClassMap[hkpConstraintInstance::NUM_PRIORITIES] = { 0, 0, 0, 0, 0, 2 };
//const hkReal hkpDefaultToiResourceMgr::s_priorityClassRatios[hkpConstraintSolverResources::NUM_PRIORITY_CLASSES] = { 0.6f, 0.0f, 0.4f };

hkResult hkpDefaultToiResourceMgr::beginToiAndSetupResources(const hkpToiEvent& event, const hkArray<hkpToiEvent>& otherEvents, hkpToiResources& resourcesOut )
{
	if (shouldHandleGivenToi(event))
	{
		resourcesOut.m_scratchpadSize = m_defaultScratchpadSize;

		HK_ASSERT3( 0xf032dede, hkMemorySystem::getInstance().solverCanAllocSingleBlock(resourcesOut.m_scratchpadSize), "Your solver buffer is too small, minimum size " << resourcesOut.m_scratchpadSize );
#ifndef HK_PLATFORM_SIM_SPU
		HK_ASSERT(0x660f3951, hkMemoryRouter::getInstance().stack().numExternalAllocations() == 0);
#endif
		m_scratchPadCapacity = resourcesOut.m_scratchpadSize;
		resourcesOut.m_scratchpad     = hkMemSolverBufAlloc<char>( m_scratchPadCapacity );

		resourcesOut.m_numToiSolverIterations            = 3; // +1 implicit pass is done incrementally in the inner loop of subSolveToi.
		resourcesOut.m_numForcedToiFinalSolverIterations = 4;

		resourcesOut.m_maxNumActiveEntities = 1000;
		resourcesOut.m_maxNumConstraints    = 1000;
		resourcesOut.m_maxNumEntities       = 1000;

		resourcesOut.m_minPriorityToProcess = hkpConstraintInstance::PRIORITY_TOI;
		
		resourcesOut.m_priorityClassMap = s_priorityClassMap;
		resourcesOut.m_priorityClassRatios = s_priorityClassRatios;

		return HK_SUCCESS;
	}
	else
	{
		return HK_FAILURE;
	}
}



void hkpDefaultToiResourceMgr::endToiAndFreeResources(const hkpToiEvent& event, const hkArray<hkpToiEvent>& otherEvents, const hkpToiResources& resources )
{
	hkMemSolverBufFree<char>( resources.m_scratchpad, m_scratchPadCapacity );
}

hkpDefaultToiResourceMgr::~hkpDefaultToiResourceMgr()
{
}


hkBool hkpDefaultToiResourceMgr::shouldHandleGivenToi( const hkpToiEvent& event )
{
	return true;
}

hkpToiResourceMgrResponse hkpDefaultToiResourceMgr::resourcesDepleted()
{
	// If you get this warning because there isn't enough space for schemas, you can
	// try adjusting the ratios of the schema buffers for the priority classes. See 
	// hkpToiResourceMgr::m_constraintPriorityRatios.
	HK_WARN(0xad000302, "TOI Resources depleted!");
	return HK_TOI_RESOURCE_MGR_RESPONSE_DO_NOT_EXPAND_AND_CONTINUE;
}

hkpToiResourceMgrResponse hkpDefaultToiResourceMgr::cannotSolve(hkArray<ConstraintViolationInfo>& violatedConstraints)
{
	//HK_WARN(0xad5D32BB, "Cannot solve constraint system in a TOI-Event.");
	return HK_TOI_RESOURCE_MGR_RESPONSE_CONTINUE;
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
