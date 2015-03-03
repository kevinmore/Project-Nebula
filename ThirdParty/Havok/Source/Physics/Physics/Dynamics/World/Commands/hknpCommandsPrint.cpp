/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Dynamics/World/Commands/hknpCommands.h>
#include <Common/Base/Container/CommandStream/hkUnrollCaseMacro.h>
#include <Common/Base/Reflection/hkClass.h>


HK_FORCE_INLINE void hknpEmptyCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
}

HK_FORCE_INLINE void hknpCreateBodyCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "createBody Id=" << m_bodyId.value() << " motionId=" << m_motionId.value();
}

HK_FORCE_INLINE void hknpDestroyBodyCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "destroyBody Id=" << m_bodyId.value();
}

HK_FORCE_INLINE void hknpAddBodyCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "addBody Id=" << m_bodyId.value() << " additionFlags=" << (int)m_additionFlags.get() << " isLastInBatch=" << m_isLastInBatch;
}

HK_FORCE_INLINE void hknpRemoveBodyCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "removeBody Id=" << m_bodyId.value() << " isLastInBatch=" << m_isLastInBatch;
}

HK_FORCE_INLINE void hknpDetachBodyCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "detachBody Id=" << m_bodyId.value() << " updateInertia=" << m_updateInertia;
}

HK_FORCE_INLINE void hknpAttachBodyCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "atachBody Id=" << m_bodyId.value() << " to " << m_compoundBodyId.value() << " updateInertia=" << m_updateInertia;
}

HK_FORCE_INLINE void hknpSetBodyTransformCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyTransform Id=" << m_bodyId.value() << " transform=" << m_transform << " activationBehavior=" << m_activationBehavior;
}

HK_FORCE_INLINE void hknpSetBodyPositionCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyPosition Id=" << m_bodyId.value() << " position=" << m_position << " activationBehavior=" << m_activationBehavior;
}

HK_FORCE_INLINE void hknpSetBodyOrientationCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyOrientation Id=" << m_bodyId.value() << " orientation=" << m_orientation << " pivot=" << m_pivot << " activationBehavior=" << m_activationBehavior;
}

HK_FORCE_INLINE void hknpSetBodyVelocityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyVelocity Id=" << m_bodyId.value() << " linearVelocity=" << m_linVelocity << " angularVelocity=" << m_angVelocity;
}

HK_FORCE_INLINE void hknpSetBodyLinearVelocityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyLinearVelocity Id=" << m_bodyId.value() << " linearVelocity=" << m_linearVelocity;
}

HK_FORCE_INLINE void hknpSetBodyAngularVelocityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyAngularVelocity Id=" << m_bodyId.value() << " angularVelocity=" << m_angularVelocity;
}

HK_FORCE_INLINE void hknpReintegrateBodyCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "reintegrateBody Id=" << m_bodyId.value() << " t=" << m_t;
}

HK_FORCE_INLINE void hknpApplyLinearImpulseCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "applyLinearImpulse Id=" << m_bodyId.value() << " impulse=" << m_impulse;
}

HK_FORCE_INLINE void hknpApplyAngularImpulseCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "applyAngularImpulse Id=" << m_bodyId.value() << " impulse=" << m_impulse;
}

HK_FORCE_INLINE void hknpApplyPointImpulseCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "applyPointImpulse Id=" << m_bodyId.value() << " impulse=" << m_impulse << " position=" << m_position;
}

HK_FORCE_INLINE void hknpSetPointVelocityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setPointVelocity Id=" << m_bodyId.value() << " velocity=" << m_velocity << " position=" << m_position;
}

HK_FORCE_INLINE void hknpSetBodyMassCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyMass Id=" << m_bodyId.value() << " massOrNegativeDensity=" << m_massOrNegativeDensity;
}

HK_FORCE_INLINE void hknpSetBodyMotionCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyMotion Id=" << m_bodyId.value() << " motionId=" << m_motionId.value();
}

HK_FORCE_INLINE void hknpSetBodyCenterOfMassCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyCenterOfMass Id=" << m_bodyId.value() << " com=" << m_com;
}

HK_FORCE_INLINE void hknpSetBodyShapeCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyShape Id=" << m_bodyId.value() << " shape=" << int(reinterpret_cast<hkUlong>(m_shape));
}

HK_FORCE_INLINE void hknpSetBodyMotionPropertiesCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyMotionProperties Id=" << m_bodyId.value() << " propertiesId=" << m_motionProperties.value();
}

HK_FORCE_INLINE void hknpSetBodyMaterialCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyMaterial Id=" << m_bodyId.value() << " materialId=" << m_material.value() << " cacheBehavior=" << m_collisionCacheBehavior;
}

HK_FORCE_INLINE void hknpSetBodyQualityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyQuality Id=" << m_bodyId.value() << " qualityId=" << m_qualityId.value() << " cacheBehavior=" << m_collisionCacheBehavior;
}

HK_FORCE_INLINE void hknpActivateBodyCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	if ( this->m_active )
	{
		out << "activateBody Id=" << m_bodyId.value();
	}
	else
	{
		out << "deactivateBody Id=" << m_bodyId.value();
	}
}

HK_FORCE_INLINE void hknpSetBodyCollisionFilterInfoCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyCollisionFilterInfo Id=" << m_bodyId.value() << " filterInfo=" << m_collisionFilterInfo;
}

HK_FORCE_INLINE void hknpRebuildBodyCollisionCachesCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "rebuildBodyCollisionCaches Id=" << m_bodyId.value();
}

HK_FORCE_INLINE void hknpSetWorldGravityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setGravity " << this->m_gravity;
}

HK_FORCE_INLINE void hknpSetMaterialFrictionCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setMaterialFriction Id=" << m_materialId.value() << " dynamicFriction=" << m_dynamicFriction;
}

HK_FORCE_INLINE void hknpSetMaterialRestitutionCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setMaterialRestitution Id=" << m_materialId.value() << " restitution=" << m_restitution;
}


HK_FORCE_INLINE void hknpSetBodyCollisionLookAheadDistanceCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "setBodyCollisionLookAheadDistance Id=" << this->m_bodyId.value() << " distance=" << m_distance << " tempExpansionVelocity=" << m_tempExpansionVelocity;
}

HK_FORCE_INLINE void hknpReserved1VelocityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "Reserved " << this->m_bodyId.value();
}

HK_FORCE_INLINE void hknpReserved2VelocityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "Reserved " << this->m_bodyId.value();
}

HK_FORCE_INLINE void hknpReserved3VelocityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "Reserved " << this->m_bodyId.value();
}

HK_FORCE_INLINE void hknpReserved4VelocityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "Reserved " << this->m_bodyId.value();
}

HK_FORCE_INLINE void hknpReserved5VelocityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "Reserved " << this->m_bodyId.value();
}

HK_FORCE_INLINE void hknpReserved6VelocityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "Reserved " << this->m_bodyId.value();
}

HK_FORCE_INLINE void hknpReserved7VelocityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "Reserved " << this->m_bodyId.value();
}

HK_FORCE_INLINE void hknpReserved8VelocityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "Reserved " << this->m_bodyId.value();
}

HK_FORCE_INLINE void hknpReserved9VelocityCommand::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "Reserved " << this->m_bodyId.value();
}


void hknpApiCommandProcessor::print( const hkCommand& command, hkOstream& stream ) const
{
#if !defined(HK_PLATFORM_CTR)
	switch (command.m_secondaryType)
	{
		HK_UNROLL_CASE_39(
				{
					typedef hkCommandTypeDiscriminator<UNROLL_I>::CommandType ct;
					const ct* c = reinterpret_cast<const ct*>( &command);
					c->printCommand(m_world, stream);
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
