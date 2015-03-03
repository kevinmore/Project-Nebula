/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulation.h>

#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobian.h>
#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.h>


void hknpSimulation::checkConsistencyOfJacobians( hknpWorld* world, hkInt32 checkFlags, hknpConstraintSolverJacobianGrid* jacGrid )
{
#if defined(HK_DEBUG)
	hknpSpaceSplitter* HK_RESTRICT splitter = world->m_spaceSplitter;

	//
	//	if the grid is a fixed grid, check for this
	//
	bool isFixed = false;
	if ( jacGrid->m_entries.getSize() == splitter->getNumCells() )
	{
		isFixed = true;
	}


	//
	//	check the link index in the cvxCacheStreams
	//
	for (int linkIndex = 0; linkIndex < jacGrid->m_entries.getSize(); linkIndex++ )
	{
		hknpConstraintSolverJacobianRange2* range = &jacGrid->m_entries[linkIndex];

		do
		{
			hknpConstraintSolverJacobianReader reader; reader.setToRange( range );

			for ( const hknpMxContactJacobian* mxJac = reader.access<hknpMxContactJacobian>(); mxJac; mxJac = reader.advanceAndAccessNext<hknpMxContactJacobian>() )
			{
				for (int k=0; k < hknpMxContactJacobian::NUM_MANIFOLDS; k++)
				{
					int solverIdA = mxJac->m_manifoldData[k].m_solverVelIdA;
					int solverIdB = mxJac->m_manifoldData[k].m_solverVelIdB;
					int cellIdxA = mxJac->m_manifoldData[k].m_cellIndexA;
					int cellIdxB = mxJac->m_manifoldData[k].m_cellIndexB;
					if ( solverIdA == 0 && solverIdB == 0 )
					{
						continue;	// empty jacobian
					}
					int checkLinkIndex = splitter->getLinkIdx( cellIdxA, cellIdxB );
					if ( isFixed )
					{
						HK_ASSERT( 0xf0edf345, cellIdxA == HKNP_INVALID_CELL_IDX || cellIdxB == HKNP_INVALID_CELL_IDX );
						checkLinkIndex = ( cellIdxA == HKNP_INVALID_CELL_IDX ) ? cellIdxB : cellIdxA;
					}
					HK_ASSERT2( 0xf034dedf, linkIndex == checkLinkIndex, "The convex cache grid sorting is messed up");
				}
			}

			range = (hknpConstraintSolverJacobianRange2*) range->m_next;
		} while(range != HK_NULL);
	}
#endif
}

void hknpSimulation::checkConsistencyOfCdCacheStream( hknpWorld* world, hkInt32 checkFlags, hknpCdCacheStream& childCdCacheStream )
{
#if defined(HK_DEBUG)
	childCdCacheStream.checkConsistency();
#endif
}

void hknpSimulation::checkConsistencyOfCdCacheGrids(
	hknpWorld* world, hkInt32 checkFlags, hkBlockStream<hknpCollisionCache>& cdCacheStream,
	hknpCdCacheStream& childCdCacheStream, hknpCdCacheGrid* grid, hknpCdCacheGrid* ppuGrid )
{
#if defined(HK_DEBUG)

	// Merge and sort ranges contained in the grids
	HK_ASSERT(0x56068460, sizeof(hkBlockStreamBase::Range) == 16);
	hknpCdCacheGrid* grids[] = { grid, ppuGrid };
	const int numGrids = ppuGrid ? 2 : 1;
	int numRanges = grid->m_entries.getSize() + (ppuGrid ? ppuGrid->m_entries.getSize() : 0);
	hkLocalBuffer<hkBlockStreamBase::Range> sortedRanges(numRanges);
	hkBlockStreamBase::Range* dst = &sortedRanges[0];
	for (int i = 0; i < numGrids; ++i)
	{
		hkArray<hkBlockStreamBase::Range>& entries = grids[i]->m_entries;
		hkString::memCpy16(dst, entries.begin(), entries.getSize());
		dst += entries.getSize();
	}
	hkSort(sortedRanges.begin(), numRanges, hkBlockStreamBase::Range::compareRange);

	// Check consistency of ranges with the cache stream
	cdCacheStream.checkConsistencyWithSortedRanges(
		sortedRanges.begin(), sortedRanges.getSize(), sizeof(hkBlockStreamBase::Range), false );

	// Check the link index in the cvxCacheStreams
	hknpSpaceSplitter* HK_RESTRICT splitter = world->m_spaceSplitter;
	hkArray<hkBlockStreamBase::Range> allCvxRanges;
	for (int i = 0; i < splitter->getNumLinks(); i++ )
	{
		for (int j = 0; j < numGrids; ++j)
		{
			hkBlockStreamBase::Range& cdCacheRange = grids[j]->m_entries[i];
			hknpCdCacheReader reader; reader.setToRange( &cdCacheRange );

			for ( const hknpCollisionCache* cdCache = reader.access(); cdCache; cdCache = reader.advanceAndAccessNext(cdCache->getSizeInBytes()) )
			{
				if ( checkFlags & CHECK_FORCE_BODIES_IN_SIMULATION )
				{
					const hknpBody& bodyA = world->getSimulatedBody( cdCache->m_bodyA );
					const hknpBody& bodyB = world->getSimulatedBody( cdCache->m_bodyB );

					const hknpMotion& motionA = world->getMotion( bodyA.m_motionId );
					const hknpMotion& motionB = world->getMotion( bodyB.m_motionId );

					int cellIdxA = motionA.m_cellIndex;
					int cellIdxB = motionB.m_cellIndex;

					if ( checkFlags & hknpSimulation::CHECK_CACHE_CELL_INDEX )
					{
						int linkIndex = splitter->getLinkIdx( cellIdxA, cellIdxB );
						HK_ASSERT2( 0xf034dedf, linkIndex == i, "The convex cache grid sorting is messed up");
					}
				}

				hknpCollisionCacheType::Enum type = cdCache->m_type;
				if ( type == hknpCollisionCacheType::CONVEX_COMPOSITE || type == hknpCollisionCacheType::COMPOSITE_COMPOSITE || type == hknpCollisionCacheType::DISTANCE_FIELD )
				{
					const hknpCompositeCollisionCache* ccCdCache = static_cast<const hknpCompositeCollisionCache*>(cdCache);
					const hkBlockStreamBase::Range& cvxRange = ccCdCache->m_childCdCacheRange;
					childCdCacheStream.checkConsistencyOfRange( cvxRange );
					allCvxRanges.pushBack(cvxRange);
				}
			}
		}
	}
	childCdCacheStream.checkConsistencyWithGrid( allCvxRanges.begin(), allCvxRanges.getSize(), sizeof(allCvxRanges[0]), false );
#endif
}


void hknpSimulation::checkConsistencyOfCdCacheStream( hknpWorld* world, hkInt32 checkFlags, hkBlockStream<hknpCollisionCache>& cdCacheStream, hknpCdCacheStream& childCdCacheStream )
{
#if defined(HK_DEBUG)
	cdCacheStream.checkConsistency();

	hkArray<hkBlockStreamBase::Range> allCvxRanges;
	//
	//	check the link index in the cvxCacheStreams
	//
	{
		hknpCdCacheReader reader; reader.setToStartOfStream( &cdCacheStream );

		for ( const hknpCollisionCache* cdCache = reader.access(); cdCache; cdCache = reader.advanceAndAccessNext(cdCache->getSizeInBytes()) )
		{
			hknpCollisionCacheType::Enum type = cdCache->m_type;
			if ( type == hknpCollisionCacheType::CONVEX_COMPOSITE || type == hknpCollisionCacheType::COMPOSITE_COMPOSITE || type == hknpCollisionCacheType::DISTANCE_FIELD )
			{
				const hknpCompositeCollisionCache* ccCdCache = static_cast<const hknpCompositeCollisionCache*>(cdCache);
				const hkBlockStreamBase::Range& childCdCacheRange = ccCdCache->m_childCdCacheRange;
				childCdCacheStream.checkConsistencyOfRange( childCdCacheRange );
				allCvxRanges.pushBack(childCdCacheRange);

				hknpCdCacheReader childReader; childReader.setToRange( &childCdCacheRange );
				int numHitsShapeKeyA = 0;

				if ( type == hknpCollisionCacheType::CONVEX_COMPOSITE )
				{
					// all caches must be convex
					for ( const hknpCollisionCache* childCache = childReader.access(); childCache; childCache = childReader.advanceAndAccessNext(childCache))
					{
						HK_ASSERT( 0xf04fffde, childCache->m_type == hknpCollisionCacheType::CONVEX_CONVEX );
					}
				}
				else if ( type == hknpCollisionCacheType::COMPOSITE_COMPOSITE  )
				{
					// all caches must be hknpChildCompositeCompositeCdCache followed by n convex
					for ( const hknpCollisionCache* childCache = childReader.access(); childCache; childCache = childReader.advanceAndAccessNext(childCache))
					{
						if ( numHitsShapeKeyA == 0)
						{
							HK_ASSERT( 0xf04fffde, childCache->m_type == hknpCollisionCacheType::SET_SHAPE_KEY_A );
							const hknpSetShapeKeyACollisionCache* childCcCdCache  = (const hknpSetShapeKeyACollisionCache*)childCache;
							numHitsShapeKeyA = childCcCdCache->m_numHitsShapeKeyA;
						}
						else
						{
							HK_ASSERT( 0xf04fffdf, childCache->m_type == hknpCollisionCacheType::CONVEX_CONVEX );
							numHitsShapeKeyA--;
						}
					}
				}
				else if ( type == hknpCollisionCacheType::DISTANCE_FIELD )
				{
					// all caches must be hknpChildCompositeCompositeCdCache followed by n convex
					for ( const hknpCollisionCache* childCache = childReader.access(); childCache; childCache = childReader.advanceAndAccessNext(childCache))
					{
						HK_ASSERT( 0xf04fffdf, childCache->m_type == hknpCollisionCacheType::CONVEX_CONVEX );
					}
				}
				HK_ASSERT( 0xfe569874, numHitsShapeKeyA == 0);
			}
		}
	}
	childCdCacheStream.checkConsistencyWithGrid( allCvxRanges.begin(), allCvxRanges.getSize(), sizeof(allCvxRanges[0]), false );



#endif
}

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
