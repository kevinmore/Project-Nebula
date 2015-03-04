/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Dynamics/World/Commands/hknpCommands.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotion.h>
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>
#include <Common/Base/Container/CommandStream/hkUnrollCaseMacro.h>


HK_FORCE_INLINE void hknpEmptyCommand::executeCommand( hknpWorld* world ) const
{
}

HK_FORCE_INLINE void hknpCreateBodyCommand::executeCommand( hknpWorld* world ) const
{
	HK_ASSERT( 0xf0345fcd, !"Not implemented");
}

HK_FORCE_INLINE void hknpAddBodyCommand::executeCommand( hknpWorld* world ) const
{
	world->addBodies( &m_bodyId, 1, m_additionFlags );
}

HK_FORCE_INLINE void hknpDestroyBodyCommand::executeCommand( hknpWorld* world ) const
{
	world->destroyBodies( &m_bodyId, 1 );
}

HK_FORCE_INLINE void hknpRemoveBodyCommand::executeCommand( hknpWorld* world ) const
{
	world->removeBodies( &m_bodyId, 1 );
}

HK_FORCE_INLINE void hknpAttachBodyCommand::executeCommand( hknpWorld* world ) const
{
	world->attachBodies( this->m_compoundBodyId, &this->m_bodyId, 1, hknpWorld::UpdateMassPropertiesMode(m_updateInertia) );
}

HK_FORCE_INLINE void hknpDetachBodyCommand::executeCommand( hknpWorld* world ) const
{
	world->detachBodies( &this->m_bodyId, 1, hknpWorld::UpdateMassPropertiesMode(m_updateInertia) );
}

HK_FORCE_INLINE void hknpSetBodyTransformCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyTransform( this->m_bodyId, this->m_transform, this->m_activationBehavior );
}

HK_FORCE_INLINE void hknpSetBodyPositionCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyPosition( this->m_bodyId, this->m_position, this->m_activationBehavior );
}

HK_FORCE_INLINE void hknpSetBodyOrientationCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyOrientation( this->m_bodyId, this->m_orientation, this->m_pivot, this->m_activationBehavior );
}

HK_FORCE_INLINE void hknpSetBodyVelocityCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyVelocity( this->m_bodyId, this->m_linVelocity, this->m_angVelocity );
}

HK_FORCE_INLINE void hknpSetBodyLinearVelocityCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyLinearVelocity( this->m_bodyId, this->m_linearVelocity );
}

HK_FORCE_INLINE void hknpSetBodyAngularVelocityCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyAngularVelocity( this->m_bodyId, this->m_angularVelocity );
}

HK_FORCE_INLINE void hknpReintegrateBodyCommand::executeCommand( hknpWorld* world ) const
{
	world->reintegrateBody( this->m_bodyId, this->m_t );
}

HK_FORCE_INLINE void hknpApplyLinearImpulseCommand::executeCommand( hknpWorld* world ) const
{
	world->applyBodyLinearImpulse( this->m_bodyId, this->m_impulse );
}

HK_FORCE_INLINE void hknpApplyAngularImpulseCommand::executeCommand( hknpWorld* world ) const
{
	world->applyBodyAngularImpulse( this->m_bodyId, this->m_impulse );
}

HK_FORCE_INLINE void hknpApplyPointImpulseCommand::executeCommand( hknpWorld* world ) const
{
	world->applyBodyImpulseAt( this->m_bodyId, this->m_impulse, this->m_position );
}

HK_FORCE_INLINE void hknpSetPointVelocityCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyVelocityAt( this->m_bodyId, this->m_velocity, this->m_position );
}

HK_FORCE_INLINE void hknpSetBodyMassCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyMass( m_bodyId, m_massOrNegativeDensity, hknpWorld::RebuildCachesMode( m_cacheBehavior) );
}

HK_FORCE_INLINE void hknpSetBodyMotionCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyMotion( m_bodyId, m_motionId, hknpWorld::RebuildCachesMode( m_cacheBehavior) );
}

HK_FORCE_INLINE void hknpSetBodyCenterOfMassCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyCenterOfMass( this->m_bodyId, this->m_com );
}

HK_FORCE_INLINE void hknpSetBodyShapeCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyShape( this->m_bodyId, this->m_shape );
}

HK_FORCE_INLINE void hknpSetBodyMotionPropertiesCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyMotionProperties( this->m_bodyId, this->m_motionProperties );
}

HK_FORCE_INLINE void hknpSetBodyMaterialCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyMaterial( this->m_bodyId, this->m_material, hknpWorld::RebuildCachesMode(this->m_collisionCacheBehavior) );
}

HK_FORCE_INLINE void hknpSetBodyQualityCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyQuality( this->m_bodyId, this->m_qualityId, hknpWorld::RebuildCachesMode(this->m_collisionCacheBehavior) );
}

HK_FORCE_INLINE void hknpActivateBodyCommand::executeCommand( hknpWorld* world ) const
{
	if ( this->m_active )
	{
		world->activateBody( this->m_bodyId );
	}
	else
	{
		world->m_deactivationManager->forceIslandOfBodyToDeactivate( this->m_bodyId );
	}
}

HK_FORCE_INLINE void hknpSetBodyCollisionFilterInfoCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyCollisionFilterInfo( this->m_bodyId, this->m_collisionFilterInfo );
}

HK_FORCE_INLINE void hknpRebuildBodyCollisionCachesCommand::executeCommand( hknpWorld* world ) const
{
	world->rebuildBodyCollisionCaches( this->m_bodyId );
}

HK_FORCE_INLINE void hknpSetWorldGravityCommand::executeCommand( hknpWorld* world ) const
{
	world->setGravity( this->m_gravity );
}

HK_FORCE_INLINE void hknpSetMaterialFrictionCommand::executeCommand( hknpWorld* world ) const
{
	hknpMaterial material = world->getMaterialLibrary()->getEntry( m_materialId );
	material.m_dynamicFriction = m_dynamicFriction;
	world->accessMaterialLibrary()->updateEntry( m_materialId, material );
}

HK_FORCE_INLINE void hknpSetMaterialRestitutionCommand::executeCommand( hknpWorld* world ) const
{
	hknpMaterial material = world->getMaterialLibrary()->getEntry( m_materialId );
	material.m_restitution = m_restitution;
	world->accessMaterialLibrary()->updateEntry( m_materialId, material );
}

HK_FORCE_INLINE void hknpSetBodyCollisionLookAheadDistanceCommand::executeCommand( hknpWorld* world ) const
{
	world->setBodyCollisionLookAheadDistance( m_bodyId, m_distance, m_tempExpansionVelocity );
}

HK_FORCE_INLINE void hknpReserved1VelocityCommand::executeCommand( hknpWorld* world ) const
{
	HK_ASSERT(0x183934a8,0); // not implemented
}
HK_FORCE_INLINE void hknpReserved2VelocityCommand::executeCommand( hknpWorld* world ) const
{
	HK_ASSERT(0x140f52e,0); // not implemented
}
HK_FORCE_INLINE void hknpReserved3VelocityCommand::executeCommand( hknpWorld* world ) const
{
	HK_ASSERT(0x24edef1b,0); // not implemented
}
HK_FORCE_INLINE void hknpReserved4VelocityCommand::executeCommand( hknpWorld* world ) const
{
	HK_ASSERT(0x653b55bc,0); // not implemented
}
HK_FORCE_INLINE void hknpReserved5VelocityCommand::executeCommand( hknpWorld* world ) const
{
	HK_ASSERT(0x5959848b,0); // not implemented
}
HK_FORCE_INLINE void hknpReserved6VelocityCommand::executeCommand( hknpWorld* world ) const
{
	HK_ASSERT(0x170346f,0); // not implemented
}
HK_FORCE_INLINE void hknpReserved7VelocityCommand::executeCommand( hknpWorld* world ) const
{
	HK_ASSERT(0x40c033f1,0); // not implemented
}
HK_FORCE_INLINE void hknpReserved8VelocityCommand::executeCommand( hknpWorld* world ) const
{
	HK_ASSERT(0x218d525e,0); // not implemented
}
HK_FORCE_INLINE void hknpReserved9VelocityCommand::executeCommand( hknpWorld* world ) const
{
	HK_ASSERT(0x46053c88,0); // not implemented
}


void hknpApiCommandProcessor::exec( const hkCommand& command )
{
	const hknpApiCommand* apiCmd = (const hknpApiCommand*)&command;
	const hknpBody& body = m_world->getBody( apiCmd->m_bodyId );
	if ( !body.isAddedToWorld() )
	{
		return;
	}

	switch (command.m_secondaryType)
	{
		HK_UNROLL_CASE_39(
				{
					typedef hkCommandTypeDiscriminator<UNROLL_I>::CommandType ct;
					const ct* c = reinterpret_cast<const ct*>(&command);
					c->executeCommand(m_world);
					break;
				}
		);
	}

	// check if our unroll macro is sufficient by checking if command 40 falls back to our empty command
	{
		typedef hkCommandTypeDiscriminator<40>::CommandType ct;
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
