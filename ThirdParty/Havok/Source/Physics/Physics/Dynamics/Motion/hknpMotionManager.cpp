/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Common/Base/Math/Vector/Mx/hkMxVector.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Physics/Physics/Dynamics/Motion/hknpMotionManager.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotion.h>
#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.h>


HK_COMPILE_TIME_ASSERT(sizeof(hknpMotion) >= sizeof(hknpMotionId));

#define REINTERPRET_AS_POINTER(motion) (*(reinterpret_cast<hknpMotionId*>(&motion)))
#define REINTERPRET_CONST_AS_POINTER(motion) (*(reinterpret_cast<hknpMotionId*>(const_cast<hknpMotion*>(&motion))))


hknpMotionManager::MotionIterator::MotionIterator( const hknpMotionManager& manager )
:	m_motionManager( manager ),
	m_index( 0 )	// start on the special static motion
{
	// Go to the first allocated motion
	next();
}


hknpMotionManager::hknpMotionManager()
:	m_firstFreeMotionId(0),
	m_firstMarkedMotionId(0),
	m_numAllocatedMotions(0),
	m_numMarkedMotions(0),
	m_peakMotionIndex(0)
{

}

void hknpMotionManager::initialize(
	hknpMotion* userMotionBuffer, int capacity,
	hknpBodyManager* bodyManager, hknpSpaceSplitter& splitter )
{
	m_bodyManager = bodyManager;

#if !defined(HK_PLATFORM_SPU)

	HK_ASSERT2( 0xb6714ad0, m_motions.isEmpty(), "Motion manager already initialized" );
	HK_ASSERT2( 0xb6714ae0, capacity > 0, "Motion buffer capacity must be > 0" );
	HK_WARN_ON_DEBUG_IF( capacity > (int)bodyManager->getCapacity(), 0xb6714ae1,
		"Motion buffer should not be larger than body buffer" );

	// allocate the motion buffer
	relocateMotionBuffer( userMotionBuffer, capacity );

	// Set up static motion 0
	m_motions[0].reset();
	m_motions[0].m_spaceSplitterWeight = 1;

	//
	const int numCells = splitter.getNumCells();
	m_activeMotionGrid.setSize( numCells );
	for( int i = 0; i < numCells; i++ )
	{
		CellData& cellData = m_activeMotionGrid[i];
		cellData.m_solverIdToMotionId.reserve( 4 * capacity / numCells );
		cellData.m_solverIdToMotionId.pushBack( hknpMotionId::STATIC );
	}

#endif	// !HK_PLATFORM_SPU

	m_firstFreeMotionId = hknpMotionId(1);
	m_numAllocatedMotions = 1;
	m_numMarkedMotions = 0;
	m_peakMotionIndex = 0;
	m_isLocked = false;
}

hknpMotionManager::~hknpMotionManager()
{
#if !defined(HK_PLATFORM_SPU)

	// deallocate the motion buffer, if owned
	if( !m_motionBufferIsUserOwned )
	{
		hkAlignedDeallocate<hknpMotion>( m_motions.begin() );
	}

#endif
}

hkBool hknpMotionManager::relocateMotionBuffer( hknpMotion* buffer, hkUint32 capacity )
{
	deleteMarkedMotions();

	const hkUint32 oldCapacity = m_motions.getCapacity();	// if zero, the buffer is being allocated for the first time

	// Check if it is possible
	{
		if( capacity == 0 )
		{
			HK_WARN( 0x4c265501, "Motion buffer capacity must be > 0. Relocation failed." );
			return false;
		}
		for( MotionIterator it(*this); it.isValid(); it.next() )
		{
			if( it.getMotionId().value() >= capacity )
			{
				HK_WARN( 0x4c265502, "Requested motion buffer capacity cannot fit existing motions. Relocation failed." );
				return false;
			}
		}
	}

	if( capacity < oldCapacity )
	{
		// Rebuild so that the free list is still walkable after shrinking
		rebuildFreeList();
	}

	//
	// Relocate the buffer
	//

	{
		const hknpMotion* bufferIn = buffer;
		hknpMotion* oldBuffer = m_motions.begin();
		HK_ASSERT( 0x3e2d2d98, (oldCapacity == 0) ^ (oldBuffer != HK_NULL) );

		// Allocate if needed
		if( !buffer )
		{
			buffer = hkAlignedAllocate<hknpMotion>( 128, capacity, HK_MEMORY_CLASS_PHYSICS );
		}

		if( oldBuffer )
		{
			// Copy/move the buffer to the new address
			const hkUint32 numToCopy = hkMath::min2( oldCapacity, capacity );
			if( !m_motionBufferIsUserOwned )
			{
				// Was owned by manager.
				hkString::memCpy( buffer, oldBuffer, numToCopy * sizeof(hknpMotion) );
				hkAlignedDeallocate<hknpMotion>( oldBuffer );
			}
			else if( buffer != oldBuffer )
			{
				// Was owned by user. Possibly overlapping.
				hkString::memMove( buffer, oldBuffer, numToCopy * sizeof(hknpMotion) );
			}
		}

		m_motions.setDataUserFree( buffer, capacity, capacity );
		m_motionBufferIsUserOwned = ( bufferIn != HK_NULL );
	}

	//
	// Fix up free list
	//

	if( capacity > oldCapacity )
	{
		// Initialize the new motions
		for( hkUint32 i=oldCapacity; i<capacity; ++i )
		{
			hknpMotion& motion = m_motions[i];
			REINTERPRET_AS_POINTER(motion) = hknpMotionId(i + 1);
			motion.m_spaceSplitterWeight = 0;
			motion.m_solverId = hknpSolverId::invalid();	// mark as invalid
		}
		REINTERPRET_AS_POINTER(m_motions[capacity-1]) = hknpMotionId(0);

		// Link the existing free list (if any)
		if( oldCapacity )
		{
			if( m_firstFreeMotionId.value() == 0 )
			{
				// Existing buffer was full
				m_firstFreeMotionId = hknpMotionId(oldCapacity);
			}
			else
			{
				// Find the end of the free list then link it
				hknpMotionId current = m_firstFreeMotionId;
				while(1)
				{
					hknpMotionId& next = REINTERPRET_AS_POINTER( m_motions[current.value()] );
					if( next.value() == 0 )
					{
						next = hknpMotionId(oldCapacity);
						break;
					}
					current = next;
				}
			}
		}
	}
	else if( capacity < oldCapacity )
	{
		// Make sure the free list is terminated
		hknpMotionId id = m_firstFreeMotionId;
		if( id.value() >= capacity )
		{
			// New buffer is full
			m_firstFreeMotionId = hknpMotionId(0);
		}
		else
		{
			// Find the end of the free list then terminate it
			while( id.value() )
			{
				hknpMotionId& nextId = REINTERPRET_AS_POINTER( m_motions[id.value()] );
				if( nextId.value() >= capacity )
				{
					nextId = hknpMotionId(0);
				}
				id = nextId;
			}
		}

		// Make sure the peak index is within bounds
		m_peakMotionIndex = hkMath::min2( m_peakMotionIndex, capacity-1 );
	}

	checkConsistency();

	// Success! Fire callbacks and return
	hknpWorld* world = m_bodyManager->m_world;
	if ( world )
	{
		world->m_signals.m_motionBufferChanged.fire(world, this);
	}
	return true;
}

void hknpMotionManager::getMotionCinfo( hknpMotionId motionId, hknpMotionCinfo& cinfoOut ) const
{
	hknpMotionCinfo* HK_RESTRICT out = &cinfoOut;
	const hknpMotion& motion = m_motions[motionId.value()];

	out->m_angularVelocity						._setRotatedDir( motion.m_orientation,motion.m_angularVelocity );
	out->m_centerOfMassWorld					= motion.getCenterOfMassInWorld();
	hkVector4 ii; motion.getInverseInertiaLocal( ii );
	out->m_inverseInertiaLocal					= ii;
	out->m_inverseMass							= ii(3);
	out->m_massFactor							= motion.getMassFactor().getReal();
	out->m_maxLinearAccelerationDistancePerStep	= motion.m_maxLinearAccelerationDistancePerStep;
	out->m_maxRotationToPreventTunneling		= motion.m_maxRotationToPreventTunneling;
	out->m_linearVelocity						= motion.m_linearVelocity;
	out->m_motionPropertiesId					= motion.m_motionPropertiesId;
	out->m_orientation							= motion.m_orientation;
}

int hknpMotionManager::calcNumActiveMotions() const
{
	int sum = 0;
	for( int i = 0; i < m_activeMotionGrid.getSize(); i++ )
	{
		const CellData& cellData = m_activeMotionGrid[i];
		sum += cellData.m_solverIdToMotionId.getSize() - 1;
	}
	return sum;
}

hknpMotionId hknpMotionManager::allocateMotion()
{
	if( m_firstFreeMotionId.value() )
	{
		hknpMotionId id = m_firstFreeMotionId;
		m_motions[id.value()].m_spaceSplitterWeight = 1;
		m_firstFreeMotionId = REINTERPRET_AS_POINTER( m_motions[id.value()] );
		m_peakMotionIndex = hkMath::max2( m_peakMotionIndex, id.value() );
		m_numAllocatedMotions++;
		return id;
	}
	else
	{
		HK_ERROR( 0x606609a5, "Motion buffer is full." );
		return hknpMotionId::invalid();
	}
}

void hknpMotionManager::deleteMarkedMotions()
{
	// If m_firstMarkedMotionId is set, so should m_numMarkedMotions be.
	HK_ASSERT( 0x814f5d0, (m_firstMarkedMotionId.value() != 0) == (m_numMarkedMotions != 0) );
	if( m_firstMarkedMotionId.value() )
	{
		// Find the end of the marked list
		hknpMotionId id = m_firstMarkedMotionId;
		HK_ON_DEBUG( hkUint32 numMarked = 0 );
		while(1)
		{
			HK_ON_DEBUG( numMarked++ );
			HK_ASSERT( 0x13f44a26, !m_motions[id.value()].isValid() );
			hknpMotionId nextId = REINTERPRET_CONST_AS_POINTER( m_motions[id.value()] );
			if( nextId.value() )
			{
				id = nextId;
			}
			else
			{
				break;
			}
		}
		HK_ASSERT( 0x814f5d0, numMarked == m_numMarkedMotions );

		// Insert the marked list at the head of the free list
		REINTERPRET_AS_POINTER( m_motions[id.value()] ) = m_firstFreeMotionId;
		m_firstFreeMotionId = m_firstMarkedMotionId;
		m_firstMarkedMotionId = hknpMotionId(0);

		m_numAllocatedMotions -= m_numMarkedMotions;
		m_numMarkedMotions = 0;
	}
}

void hknpMotionManager::rebuildFreeList()
{
	// Gather free IDs
	hkLocalBuffer<hknpMotionId> freeIds( getCapacity() - getNumAllocatedMotions() );
	int numFreeIds = 0;
	{
		hknpMotionId id = m_firstFreeMotionId;
		while( id.value() )
		{
			freeIds[numFreeIds++] = id;
			id = REINTERPRET_CONST_AS_POINTER( m_motions[id.value()] );
		}
	}

	// Sort
	hkSort( freeIds.begin(), numFreeIds );

	// Rebuild
	m_firstFreeMotionId = hknpMotionId(0);
	for( int i = numFreeIds-1; i >= 0; --i )
	{
		hknpMotionId id = freeIds[i];
		REINTERPRET_AS_POINTER(m_motions[id.value()]) = m_firstFreeMotionId;
		m_firstFreeMotionId = id;
	}
}



void hknpMotionManager::initializeMotion( hknpMotion* HK_RESTRICT motion, const hknpMotionCinfo& motionCinfo, const hknpSpaceSplitter& spaceSplitter ) const
{
	motion->setCenterOfMassInWorld( motionCinfo.m_centerOfMassWorld );
	motion->m_orientation = motionCinfo.m_orientation;

	hkSimdReal invMass; invMass.setFromFloat(motionCinfo.m_inverseMass);
	hkVector4 invInertia; invInertia.setXYZ_W(motionCinfo.m_inverseInertiaLocal, invMass );
	invInertia.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(motion->m_inverseInertia);
	motion->setMassFactor( hkSimdReal::fromFloat(motionCinfo.m_massFactor) );
	motion->m_maxLinearAccelerationDistancePerStep.setReal<true>(motionCinfo.m_maxLinearAccelerationDistancePerStep);

	if ( motionCinfo.m_motionPropertiesId.isValid() )
	{
		motion->m_motionPropertiesId = motionCinfo.m_motionPropertiesId;
	}
	else
	{
		motion->m_motionPropertiesId = hknpMotionPropertiesId::DYNAMIC;
	}

	motion->m_linearVelocity = motionCinfo.m_linearVelocity;
	motion->m_angularVelocity.setRotatedInverseDir( motion->m_orientation, motionCinfo.m_angularVelocity );

	motion->m_maxRotationToPreventTunneling.setReal<true>(motionCinfo.m_maxRotationToPreventTunneling);
	motion->m_previousStepLinearVelocity.setZero();
	motion->m_previousStepAngularVelocity.setZero();

	motion->m_linearVelocityCage[0].setZero();
	motion->m_linearVelocityCage[1].setZero();
	motion->m_linearVelocityCage[2].setZero();

	motion->m_integrationFactor.setZero();

	motion->m_firstAttachedBodyId = hknpBodyId::invalid();
	motion->m_cellIndex = hknpCellIndex( spaceSplitter.getCellIdx( motion->getCenterOfMassInWorld() ) );
	motion->m_solverId = hknpSolverId::invalid();

	motion->m_spaceSplitterWeight = 1;
}

void hknpMotionManager::addActiveMotion( hknpMotion& motion, hknpMotionId motionId )
{
	HK_ASSERT2( 0xf0334ed4, !motion.m_solverId.isValid(), "Your motion is already activated" );
	HK_ASSERT2( 0xf0334ed5, motion.m_cellIndex != HKNP_INVALID_CELL_IDX, "Your motion has no cell index yet" );

	//
	//	Update histogram
	//	And get a solver ID
	//
	int cellIndex = motion.m_cellIndex;
	CellData& cellData = m_activeMotionGrid[cellIndex];

	int solverId = cellData.m_solverIdToMotionId.getSize();
	HK_ASSERT( 0xf0345456, solverId > 0 ); // because of fixed motions
	motion.m_solverId = hknpSolverId( solverId );
	cellData.m_solverIdToMotionId.pushBack( motionId );
}

void hknpMotionManager::removeActiveMotion( hknpMotion& motion, hknpMotionId motionId )
{
	HK_ASSERT2( 0xf0334ed6, motion.m_solverId.isValid(), "Your motion is already inactive" );
	HK_ASSERT2( 0xf0334ed7, motion.m_cellIndex != HKNP_INVALID_CELL_IDX, "Your motion has no cell index yet" );
	HK_ASSERT( 0xf0334ed8, !m_isLocked );

	//
	//	And get me my solver Id
	//
	int cellIndex = motion.m_cellIndex;
	hknpSolverId solverId  = motion.m_solverId;

	//
	// Now we need to move the motion at the end of the m_solverIdToMotion
	//
	CellData& cellData = m_activeMotionGrid[cellIndex];

	// check if we are really registered at that cell
	HK_ASSERT( 0xf0334ed9, cellData.m_solverIdToMotionId[solverId.value()] == motionId );

	hkUint32 otherSolverId = cellData.m_solverIdToMotionId.getSize() - 1;
	if ( solverId.value() < otherSolverId )
	{
		hknpMotionId otherMotionId = cellData.m_solverIdToMotionId[ otherSolverId ];
		hknpMotion& otherMotion = m_motions[ otherMotionId.value() ];
		otherMotion.m_solverId = solverId;
		cellData.m_solverIdToMotionId[ solverId.value() ] = otherMotionId;
	}
	motion.m_solverId = hknpSolverId::invalid();
	cellData.m_solverIdToMotionId.popBack();
}

void hknpMotionManager::buildSolverIdToMotionIdMap(
	hknpIdxRangeGrid& cellIdxToGlobalSolverIdOut, hkArray<hknpMotionId>& solverIdToMotionIdOut )
{
	int numTotalActiveMotions = calcNumActiveMotions();
	int numCells = getNumCells();

	HK_ASSERT2(0xad234a2, solverIdToMotionIdOut.isEmpty(), "The output array is not empty.");
	HK_ASSERT2(0xad234a4, cellIdxToGlobalSolverIdOut.m_entries.getSize() == numCells,   "Motion grid wrong dimension.");

	solverIdToMotionIdOut.reserve( numTotalActiveMotions + 4 * numCells ); // reserve with padding

	// build prefix sum and motionGrid
	hkUint32 sum = 0;
	for( int cellIndex=0; cellIndex<numCells; cellIndex++ )
	{
		const hkArray<hknpMotionId>& activeMotions = getSolverIdToMotionIdForCell( cellIndex );
		int numActiveInCell = activeMotions.getSize();
		if( numActiveInCell > 1 ) // save memory by sharing the special motion of empty cell with the next cell
		{
			int numActiveInCellPadded = HK_NEXT_MULTIPLE_OF(hk4xVector4::mxLength, numActiveInCell);
			cellIdxToGlobalSolverIdOut.m_entries[cellIndex] = hknpIdxRange(sum, numActiveInCell);

			//
			// convert cell local solverIdToMotionId to global solverIdToMotionId
			//
			hknpMotionId* idsDest = solverIdToMotionIdOut.expandByUnchecked( numActiveInCellPadded );
			idsDest[0] = hknpMotionId::STATIC;
			HK_ASSERT( 0xf0345456, activeMotions[0] == hknpMotionId::STATIC );
			int i;
			for ( i=1; i < numActiveInCell; i++)
			{
				idsDest[i] = activeMotions[i];
			}
			// add padding
			for ( ; i < numActiveInCellPadded; i++)
			{
				idsDest[i] = hknpMotionId::STATIC;
			}
			sum += numActiveInCellPadded;
		}
		else
		{
			// empty cell simply invalid motion.
			// Use sum as start index to keep the grid sorted by start index even with presence of
			// empty ranges, this enables binary searching on the grid. (see the function findCellIdx)
			cellIdxToGlobalSolverIdOut.m_entries[cellIndex] = hknpIdxRange( sum, 0 );
		}
	}

#if defined(HK_DEBUG)
	for( int solverId=0; solverId<solverIdToMotionIdOut.getSize(); ++solverId )
	{
		hknpMotionId motionId = solverIdToMotionIdOut[solverId];
		HK_ASSERT2(0x39f1ad2a, motionId.isValid() , "incomplete s->m mapping");
		const hknpMotion& m = m_motions[motionId.value()]; // will fail if not exist
		if( motionId != hknpMotionId::STATIC )
		{
			int cellIndex = m.m_cellIndex;
			HK_ASSERT( 0x15e1defe, cellIndex != hknpCellIndex(HKNP_INVALID_CELL_IDX) );
			int globalSolverId = m.m_solverId.value() + cellIdxToGlobalSolverIdOut.m_entries[ cellIndex ].m_start;
			HK_ASSERT( 0x15e1defe, globalSolverId == solverId );
		}
	}
#endif
}

void hknpMotionManager::rebuildMotionHistogram( const hknpSpaceSplitter& spaceSplitter )
{
	hkArray<CellData> newCellData;
	newCellData.setSize( m_activeMotionGrid.getSize() );

	for (int i = 0; i < newCellData.getSize(); i++)
	{
		CellData& cellData = newCellData[i];
		cellData.m_solverIdToMotionId.pushBack(hknpMotionId::STATIC);
	}

	for (int motionId = 1; motionId < m_motions.getSize(); motionId++)	// skip fixed motion
	{
		hknpMotion& motion = m_motions[motionId];
		hknpSolverId solverId = motion.m_solverId;
		if ( motion.isActive() )
		{
			int cellIdx = spaceSplitter.getCellIdx( motion.getCenterOfMassInWorld() );
			CellData& cellData = newCellData[cellIdx];
			solverId = hknpSolverId( cellData.m_solverIdToMotionId.getSize() );
			cellData.m_solverIdToMotionId.pushBack( hknpMotionId(motionId) );

			motion.m_solverId = solverId;
			if ( motion.m_cellIndex != hknpCellIndex(cellIdx))
			{
				m_bodyManager->updateBodyToCellIndexTable( motion.m_firstAttachedBodyId, hknpCellIndex(cellIdx) );
				motion.m_cellIndex = hknpCellIndex(cellIdx);
			}
		}
	}

	m_activeMotionGrid.swap(newCellData);
	m_bodyManager->checkConsistency();
}

void hknpMotionManager::checkConsistency()
{
#if defined(HK_DEBUG)

	HK_TIME_CODE_BLOCK( "CheckMotionManagerConsistency", HK_NULL );

	// Check motion buffer consistency
	if( m_numAllocatedMotions )	// skip during manager initialization
	{
		HK_ASSERT( 0xf0dfede5, m_motions.getSize() == m_motions.getCapacity() );
		HK_ASSERT( 0x3bc15dd6, getNumAllocatedMotions() <= getCapacity() );
		HK_ASSERT( 0x3bc15dd7, m_numMarkedMotions < m_numAllocatedMotions );

		// Gather "free" motions
		hkUint32 numFree = 0;
		hkLocalBitField freeMotions( m_motions.getSize(), hkBitFieldValue::ZERO );
		{
			hknpMotionId id = m_firstFreeMotionId;
			while( id.value() )
			{
				numFree++;
				freeMotions.set( id.value() );
				id = REINTERPRET_CONST_AS_POINTER( m_motions[id.value()] );
			}
		}
		HK_ASSERT( 0x3bc15dd8, numFree + m_numAllocatedMotions == getCapacity() );

		// Gather "marked for deletion" motions
		hkUint32 numMarked = 0;
		hkLocalBitField markedMotions( m_motions.getSize(), hkBitFieldValue::ZERO );
		{
			hknpMotionId id = m_firstMarkedMotionId;
			while( id.value() )
			{
				numMarked++;
				markedMotions.set( id.value() );
				id = REINTERPRET_CONST_AS_POINTER( m_motions[id.value()] );
			}
		}
		HK_ASSERT( 0x3bc15dd9, numMarked == m_numMarkedMotions );

		hkBitField anded;
		anded = freeMotions;
		anded.andWith( markedMotions );
		HK_ASSERT2( 0x3bc15dda, !anded.anyIsSet(), "Some bodies are both marked for deletion and free!" );

		// Check consistency of hknpMotion::m_spaceSplitterWeight
		{
			hkBitField ored;
			ored = freeMotions;
			ored.orWith(markedMotions);
			for (hkBitField::Iterator iterator(ored); iterator.isValid(ored); iterator.getNext(ored))
			{
				const hknpMotion& motion = m_motions[iterator.getCurrentBit()];
				HK_ASSERT(0x3db02159, iterator.isCurrentBitSet() != motion.isValid());
			}
		}
	}

	//
	if( !m_activeMotionGrid.isEmpty() )
	{
		hkArray<int> motionHistogram;
		motionHistogram.setSize( m_activeMotionGrid.getSize(), 0 );

		for( hkUint32 i = 1; i < (hkUint32)m_motions.getSize(); i++ )	// skip fixed motion
		{
			const hknpMotion& motion = m_motions[i];
			if( motion.isActive() )
			{
				const int cellIdx = motion.m_cellIndex;
				const hknpSolverId solverId = motion.m_solverId;

				const CellData& cellData = m_activeMotionGrid[cellIdx];
				HK_ASSERT( 0xf034ed45, cellData.m_solverIdToMotionId[solverId.value()].value() == i );
				motionHistogram[cellIdx]++;
			}
		}
		for( int i = 0; i < m_activeMotionGrid.getSize(); i++ )
		{
			const CellData& cellData = m_activeMotionGrid[i];
			HK_ASSERT( 0xf034ed45, motionHistogram[i] == cellData.m_solverIdToMotionId.getSize() - 1 );
		}
	}

#endif	// HK_DEBUG
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
