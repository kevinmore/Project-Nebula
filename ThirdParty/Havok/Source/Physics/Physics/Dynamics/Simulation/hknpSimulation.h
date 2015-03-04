/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SIMULATION_H
#define HKNP_SIMULATION_H

#include <Physics/Physics/Dynamics/Simulation/hknpSimulationContext.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>

class hknpConstraintSolverJacobianGrid;
class hknpCdCacheStream;
class hknpSolverData;


/// Abstract class for stepping a simulation world.
class hknpSimulation
{
	public:

		enum CheckConsistencyFlags
		{
			CHECK_CACHES = 1<<0,						///< check collision caches
			CHECK_CACHE_CELL_INDEX = 1<<1,				///< check if the body cell index matches the cache cell index.
			CHECK_FORCE_BODIES_IN_SIMULATION = 1<<2,	///< if set all bodies referenced by caches must be in simulation.
			CHECK_ALL = ~0
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpSimulation );

		virtual ~hknpSimulation() {}

		/// Perform collision detection and constraint setup.
		/// This allocates and populates solverDataOut.
		/// If simulationContext->m_taskGraph contains any tasks after this is called, the task graph must be processed
		/// and this method must be called again.
		virtual void collide( hknpSimulationContext& simulationContext, hknpSolverData*& solverDataOut ) = 0;

		/// Perform solving and integration.
		/// If simulationContext->m_taskGraph contains any tasks after this is called, the task graph must be processed
		/// and this method must be called again.
		virtual void solve( hknpSimulationContext& simulationContext, hknpSolverData* solverData ) = 0;

		/// Check consistency
		virtual void checkConsistency( hknpWorld* world, hkInt32 checkFlags ) {}

	public:

		//
		// Consistency checks
		//

		static void HK_CALL checkConsistencyOfJacobians(
			hknpWorld* world, hkInt32 checkFlags, hknpConstraintSolverJacobianGrid* jacGrid );

		static void HK_CALL checkConsistencyOfCdCacheStream(
			hknpWorld* world, hkInt32 checkFlags, hknpCdCacheStream& childCdCacheStream );

		static void HK_CALL checkConsistencyOfCdCacheGrids(
			hknpWorld* world, hkInt32 checkFlags, hkBlockStream<hknpCollisionCache>& cdCacheStream,
			hknpCdCacheStream& childCdCacheStream, hknpCdCacheGrid* grid, hknpCdCacheGrid* ppuGrid = HK_NULL);

		static void HK_CALL checkConsistencyOfCdCacheStream(
			hknpWorld* world, hkInt32 checkFlags, hkBlockStream<hknpCollisionCache>& cdCacheStream,
			hknpCdCacheStream& childCdCacheStream );
};


#endif // HKNP_SIMULATION_H

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
