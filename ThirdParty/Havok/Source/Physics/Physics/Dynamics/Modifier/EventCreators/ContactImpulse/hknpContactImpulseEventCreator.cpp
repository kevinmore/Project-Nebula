/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Modifier/EventCreators/ContactImpulse/hknpContactImpulseEventCreator.h>

#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Dynamics/World/Events/hknpEvents.h>
#include <Physics/Physics/Dynamics/Solver/Event/hknpSolverEvents.h>

#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>
#include <Physics/Physics/Collide/hknpCdBody.h>

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobian.h>


void hknpContactImpulseEventCreator::postContactSolve(
	const hknpSimulationThreadContext& tl, const hknpModifier::SolverCallbackInput& input,
	hkVector4Parameter contactImpulses, hkReal frictionFactor )
{
	const hknpMxContactJacobian::ManifoldData& manifoldData = input.m_contactJacobian->m_manifoldData[input.m_manifoldIndex];
	if( manifoldData.m_manifoldType == hknpManifold::TYPE_NORMAL )
	{
		const hknpManifoldSolverInfo* msi = &(input.m_collisionCache->m_manifoldSolverInfo);
		// Get current impulse state.
		const int haveImpulses = ( 0 != contactImpulses.greaterZero().anyIsSet() );

		// Get previous impulse state. (No need to decompress, just check if they are not zero.)
		HK_ASSERT( 0x633f3483, msi );
		const int hadImpulses = ( 0 != hkString::memCmp( msi->m_impulses, &hkVector4::getZero(), sizeof(msi->m_impulses) ) );

		// Convert to status.
		HK_COMPILE_TIME_ASSERT( hknpContactImpulseEvent::STATUS_NONE		== 0 );
		HK_COMPILE_TIME_ASSERT( hknpContactImpulseEvent::STATUS_STARTED		== 1 );
		HK_COMPILE_TIME_ASSERT( hknpContactImpulseEvent::STATUS_FINISHED	== 2 );
		HK_COMPILE_TIME_ASSERT( hknpContactImpulseEvent::STATUS_CONTINUED	== 3 );
		const hknpContactImpulseEvent::Status status = (hknpContactImpulseEvent::Status)( haveImpulses | (hadImpulses<<1) );

		// Filter
		if	( ( status == hknpContactImpulseEvent::STATUS_NONE ) ||
			( ( status == hknpContactImpulseEvent::STATUS_CONTINUED ) && !( msi->m_flags.get( hknpManifoldSolverInfo::RAISE_CONTACT_IMPULSE_CONTINUED_EVENTS ) ) ) ||
			( ( status == hknpContactImpulseEvent::STATUS_FINISHED ) && !( msi->m_flags.get( hknpManifoldSolverInfo::RAISE_CONTACT_IMPULSE_FINISHED_EVENTS ) ) ) )
		{
			return;
		}
		else if( status == hknpContactImpulseEvent::STATUS_STARTED )
		{
			// Reset filter flags
			msi->m_flags.clear(
				hknpManifoldSolverInfo::RAISE_CONTACT_IMPULSE_CONTINUED_EVENTS |
				hknpManifoldSolverInfo::RAISE_CONTACT_IMPULSE_FINISHED_EVENTS );
		}

		// Create event
		{
			hknpContactImpulseEvent event( manifoldData.m_bodyIdA, manifoldData.m_bodyIdB );
			event.initialize( input );
			event.m_status = status;
			event.m_frictionFactor = frictionFactor;
			contactImpulses.store<4>( &event.m_contactImpulses[0] );

			tl.execCommand( event );
		}
	}
}

void hknpContactImpulseEventCreator::manifoldDestroyedCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
	hknpManifoldCollisionCache* HK_RESTRICT cache, hknpCdCacheDestructReason::Enum reason)
{
	if( cache->m_manifoldSolverInfo.m_flags.get( hknpManifoldSolverInfo::RAISE_CONTACT_IMPULSE_FINISHED_EVENTS ) )
	{
		// Get previous impulse state. (No need to decompress, just check if they are not zero.)
		const bool hadImpulses = ( 0 != hkString::memCmp(
			cache->m_manifoldSolverInfo.m_impulses, &hkVector4::getZero(),
			sizeof(cache->m_manifoldSolverInfo.m_impulses) ) );

		if( hadImpulses )
		{
			hknpContactImpulseEvent event( cdBodyA.m_body->m_id, cdBodyB.m_body->m_id );
			event.m_status = hknpContactImpulseEvent::STATUS_FINISHED;
			event.m_frictionFactor = 0.0f;
			event.m_contactImpulses[0] = 0.0f;
			event.m_contactImpulses[1] = 0.0f;
			event.m_contactImpulses[2] = 0.0f;
			event.m_contactImpulses[3] = 0.0f;

			tl.execCommand(event);
		}
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
