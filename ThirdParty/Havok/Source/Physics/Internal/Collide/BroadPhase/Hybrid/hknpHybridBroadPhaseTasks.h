/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_HYBRID_BROAD_PHASE_TASKS_H
#define HKNP_HYBRID_BROAD_PHASE_TASKS_H

#include <Common/Base/Thread/Task/hkTask.h>
#include <Physics/Internal/Collide/BroadPhase/Hybrid/hknpHybridBroadPhase.h>

// Hybrid broad phase tasks
class hknpHBPUpdateDirtyBodiesTask;
class hknpHBPUpdateTreeTask;
class hknpHBPTreeVsTreeTask;
class hknpHBPFinishUpdateTask;

// A container for hybrid broad phase tasks.
class hknpHybridBroadPhaseTaskContext
{
	public:

		enum
		{
			MAX_COLLIDE_TREE_PAIRS = 16 
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpHybridBroadPhaseTaskContext );

		hknpHybridBroadPhaseTaskContext();
		~hknpHybridBroadPhaseTaskContext();

		void buildTaskGraph(
			hknpWorld* world, hknpSimulationContext* simulationContext,
			hkBlockStream<hknpBodyIdPair>* newPairsStream, hkTaskGraph* taskGraphOut );

	public:

		// Context shared between the tasks
		hknpSimulationContext*	m_simulationContext;
		hknpHybridBroadPhase*	m_broadPhase;
		hknpBody*				m_bodies;
		const hkAabb16*			m_previousAabbs;
		hknpBodyManager*		m_bodyManager;

		// Reusable tasks
		hknpHBPUpdateDirtyBodiesTask*	m_updateDirtyBodiesTask;
		hknpHBPUpdateTreeTask*			m_updateTreeTaskPool[hknpHybridBroadPhase::s_maxNumTrees];
		hknpHBPTreeVsTreeTask*			m_treeVsTreeTaskPool[MAX_COLLIDE_TREE_PAIRS];
		hknpHBPFinishUpdateTask*		m_finishUpdateTask;

		// The stream where new pairs should be written to
		hkBlockStream<hknpBodyIdPair>*	m_newPairsStream;
};

#endif // HKNP_HYBRID_BROAD_PHASE_TASKS_H

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
