/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/World/Commands/hknpInternalCommands.h>

#include <Common/Base/Container/CommandStream/hkUnrollCaseMacro.h>

#include <Physics/Physics/Collide/Event/hknpCollideEvents.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>

#include <Physics/Physics/Dynamics/Solver/Event/hknpSolverEvents.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverData.h>
#include <Physics/Physics/Dynamics/World/Events/hknpEventDispatcher.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Dynamics/Modifier/TriggerVolume/hknpTriggerVolumeModifier.h>

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobian.h>


void hknpInternalCommandProcessor::exec( const hkCommand& command )
{
	const hknpInternalCommand* apiCmd = (const hknpInternalCommand*)&command;

	// The body might have been destroyed by earlier command handlers
	if( HK_VERY_UNLIKELY( !m_world->isBodyValid( apiCmd->m_bodyId ) ) )
	{
		return;
	}

	const hknpBody& body = m_world->getBody( apiCmd->m_bodyId );

	switch (apiCmd->m_secondaryType)
	{
	case hknpInternalCommand::CMD_CELL_INDEX_MODIFIED:
		{
			const hknpCellIndexModifiedCommand* c = (const hknpCellIndexModifiedCommand*)apiCmd;
			if ( !body.isAddedToWorld() )
			{
				return;
			}

			hknpMotionId motionId = body.m_motionId;
			hknpMotion& motion = m_world->accessMotionUnchecked( motionId );
			m_world->m_motionManager.updateCellIdx( motion, motionId, c->m_newCellIndex );

			break;
		}

	case hknpInternalCommand::CMD_ATOM_SOLVER_FORCE_CLIPPED:
		{
			//const hknpAtomSolverForceClippedCommand* c = (const hknpAtomSolverForceClippedCommand*)apiCmd;
			break;
		}

	case hknpInternalCommand::CMD_VALIDATE_TRIGGER_VOLUME_EVENT:
		{
			const hknpValidateTriggerVolumeEventCommand* c = (const hknpValidateTriggerVolumeEventCommand*)apiCmd;
			if ( !body.isAddedToWorld() )
			{
				return;
			}

			// Check if the bodies actually penetrated
			const hkSimdReal toi = c->calculateToi( m_world, hkSimdReal_0 );
			if( toi < hkSimdReal_1 )
			{
				const hknpContactJacobianTypes::HeaderData& manifoldData = c->getJacobianHeader();
				HK_ASSERT(0x2306b4ef, manifoldData.m_collisionCacheInMainMemory &&
					!manifoldData.m_collisionCacheInMainMemory->m_manifoldSolverInfo.m_flags.get( hknpManifoldSolverInfo::WAS_TRIGGER_VOLUME_ENTERED ) );

				// Create a trigger volume entered event
				hknpTriggerVolumeEvent tvEvent(
					c->m_bodyIds[0], manifoldData.m_shapeKeyA,
					c->m_bodyIds[1], manifoldData.m_shapeKeyB,
					hknpTriggerVolumeEvent::STATUS_ENTERED );

				// Dispatch it
				
				m_world->getEventDispatcher()->exec( tvEvent );

				manifoldData.m_collisionCacheInMainMemory->m_manifoldSolverInfo.m_flags.orWith( hknpManifoldSolverInfo::WAS_TRIGGER_VOLUME_ENTERED );
			}

			break;
		}

	case hknpInternalCommand::CMD_MOTION_WELD_TOI:
		{
			const hknpMotionWeldTOICommand* c = (const hknpMotionWeldTOICommand*)apiCmd;
			if ( !body.isAddedToWorld() )
			{
				return;
			}

			HK_ON_DEBUG( const int a = HK_OFFSET_OF(hknpInternalCommand, m_bodyId) );
			HK_ON_DEBUG( const int b = HK_OFFSET_OF(hknpMotionWeldTOICommand, m_bodyIds[0]) );
			HK_ASSERT( 0xf03ded4d, a == b );

			// Calculate time of impact
			hkSimdReal toi;
			{
				const hknpMaterial& material = m_world->getMaterialLibrary()->getEntry( body.m_materialId );
				hkSimdReal maxExtraPenetration; maxExtraPenetration.setFromHalf( material.m_weldingTolerance );
				toi = c->calculateToi( m_world, maxExtraPenetration );
			}

			// Move the body to the point of impact
			if ( toi < hkSimdReal_1 )
			{
				if ( !body.isStaticOrKeyframed() )
				{
					m_world->reintegrateBody( apiCmd->m_bodyId, toi.getReal() );
				}
				if( m_world->isBodyAdded( c->m_bodyIds[1] ) )
				{
					m_world->reintegrateBody( c->m_bodyIds[1], toi.getReal() );
				}
			}

			break;
		}

	case hknpInternalCommand::CMD_ADD_CONSTRAINT_RANGE:
		{
			hknpAddConstraintRangeCommand* c = (hknpAddConstraintRangeCommand*)apiCmd;

			// lets pick a random stream:
			hknpConstraintSolverJacobianStream* stream = &m_solverData->m_threadData[0].m_jacConstraintsStream;

			hknpConstraintSolverJacobianWriter jacWriter; jacWriter.setToEndOfStream( m_simulationThreadContext->m_tempAllocator, stream );

			c->m_grid->addRange( jacWriter, c->m_linkIndex, c->m_range );
			jacWriter.finalize();

			break;
		}

	default:
		break;
	}
}

HK_FORCE_INLINE void hknpCellIndexModifiedCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "CellIndexModifiedCommand Id=" << m_bodyId.value() << " OldCellIndex=" << int(m_oldCellIndex) << " NewCellIndex="
		<< int(m_newCellIndex);
}

HK_FORCE_INLINE void hknpAtomSolverForceClippedCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "hknpAtomSolverForceClippedCommand factor=" << m_impulseFactor;
}

HK_FORCE_INLINE void hknpValidateTriggerVolumeEventCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "hknpValidateTriggerVolumeEventCommand id[0]=" << m_bodyIds[0].value() << " id[1]=" << m_bodyIds[1].value();
}


void hknpMotionWeldTOICommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	hkcdManifold4 manifold;
	calculateContactPointPositions( world, &manifold );
	out << "hknpMotionWeldTOICommand bodyIds=" << m_bodyIds[0].value() << "," << m_bodyIds[1].value() << " normal=" << manifold.m_normal
		<< " pos0=" << manifold.m_positions[0]
	<< " pos1=" << manifold.m_positions[1]
	<< " pos2=" << manifold.m_positions[2]
	<< " pos3=" << manifold.m_positions[3]
	
		;
}

void hknpAddConstraintRangeCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "hknpAddConstraintRangeCommand idA=" << m_bodyId.value();
}


void hknpInternalCommandProcessor::print( const hkCommand& command, hkOstream& stream ) const
{
// 	const hknpInternalCommand* apiCmd = (const hknpInternalCommand*)&command;
// 	hknpBody& body = m_world->m_bodyManager.getBodies().begin()[ apiCmd->m_bodyId.value() ];
// 	if ( !body.isValid() )
// 	{
// 		return;
// 	}

	switch (command.m_secondaryType)
	{
		HK_UNROLL_CASE_08(
		{
			typedef hknpInternalCommandTypeDiscriminator<UNROLL_I>::CommandType ct;
			const ct* c = reinterpret_cast<const ct*>(&command);
			c->printCommand(m_world, stream );
			break;
		}
		);
	}

	// check if our unroll macro is sufficient by checking if command 33 falls back to our empty command
	{
		typedef hknpInternalCommandTypeDiscriminator<9>::CommandType ct;
		const ct* c = reinterpret_cast<const ct*>(&command);
		c->checkIsEmptyCommand();
	}
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
