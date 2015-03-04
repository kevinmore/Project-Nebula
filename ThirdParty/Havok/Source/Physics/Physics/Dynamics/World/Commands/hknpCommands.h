/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COMMANDS_H
#define HKNP_COMMANDS_H

#include <Common/Base/Container/CommandStream/hkCommandStream.h>


/// Base class for all API commands to the physics engine.
/// In debug you can add a trace to hknpWorld, which will generate a log of hknpWorld API calls,
/// by sending hknpApiCommands to a stream.
class hknpApiCommand : public hkCommand
{
	public:

		enum SecondaryType
		{
			CMD_CREATE_BODY,
			CMD_ADD_BODY,
			CMD_DESTROY_BODY,
			CMD_REMOVE_BODY,
			CMD_ATTACH_BODY,
			CMD_DETACH_BODY,
			CMD_SET_BODY_TRANSFORM,
			CMD_SET_BODY_POSITION,
			CMD_SET_BODY_ORIENTATION,
			CMD_SET_BODY_VELOCITY,
			CMD_SET_BODY_LINEAR_VELOCITY,
			CMD_SET_BODY_ANGULAR_VELOCITY,
			CMD_APPLY_LINEAR_IMPULSE,
			CMD_APPLY_ANGULAR_IMPULSE,
			CMD_APPLY_POINT_IMPULSE,
			CMD_SET_POINT_VELOCITY,
			CMD_REINTEGRATE_BODY,

			CMD_SET_BODY_MOTION,
			CMD_SET_BODY_MASS,
			CMD_SET_BODY_CENTER_OF_MASS,
			CMD_SET_BODY_SHAPE,
			CMD_SET_BODY_MOTION_PROPERTIES,
			CMD_SET_BODY_MATERIAL,
			CMD_SET_BODY_QUALITY,
			CMD_ACTIVATE_BODY,
			CMD_SET_BODY_FILTER_INFO,
			CMD_REBUILD_BODY_COLLISION_CACHES,

			CMD_SET_BODY_COLLISION_LOOKAHEAD_DISTANCE,
			CMD_RESERVED_1,
			CMD_RESERVED_2,
			CMD_RESERVED_3,
			CMD_RESERVED_4,
			CMD_RESERVED_5,
			CMD_RESERVED_6,
			CMD_RESERVED_7,
			CMD_RESERVED_8,
			CMD_RESERVED_9,

			CMD_SET_WORLD_GRAVITY,
			CMD_SET_MATERIAL_FRICTION,
			CMD_SET_MATERIAL_RESTITUTION,
			CMD_MAX
		};

	public:

		hknpApiCommand( hknpBodyId bodyId, hkUint16 subType, int sizeInBytes ) :
			hkCommand( TYPE_PHYSICS_API, subType, sizeInBytes ),
			m_bodyId( bodyId ) {}

	public:

		hknpBodyId m_bodyId;
};


// A list of default command functions to be used in the header of a command (non virtual).
#define HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS \
	void executeCommand( hknpWorld* world ) const;	\
	void printCommand( hknpWorld* world, hkOstream& stream) const;


/// Empty command, needed to debug the dispatcher.
struct hknpEmptyCommand : public hknpApiCommand
{
	hknpEmptyCommand( hknpBodyId id ) : hknpApiCommand( id, CMD_DESTROY_BODY, sizeof(*this) ) {}
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;
	void checkIsEmptyCommand() const {}	/// This allows the compiler to check that all commands are dispatched
};


// Helper structures to do allow for implementing command dispatching without vtables.
// Check out hknpApiCommandProcessor::exec() for how to do this.
#define HK_DECLARE_COMMAND_DISCRIMINATOR( TYPE, ID)	\
template <     >	struct hkCommandTypeDiscriminator<hknpApiCommand::ID>{ typedef TYPE CommandType; }

template <int X>	struct hkCommandTypeDiscriminator					 { typedef hknpEmptyCommand CommandType; };


/// A command processor dispatching physics commands.
class hknpApiCommandProcessor : public hkSecondaryCommandDispatcher
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE );

		/// Constructor.
		HK_FORCE_INLINE hknpApiCommandProcessor( hknpWorld* world ) : m_world(world) {}

		/// Dispatch commands.
		virtual void exec( const hkCommand& command );

		/// Print.
		virtual void print( const hkCommand& command, hkOstream& stream ) const;

	public:

		/// Backlink to the world.
		hknpWorld* m_world;
};


/// Create body command.
struct hknpCreateBodyCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpCreateBodyCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpCreateBodyCommand( hknpBodyId id, hknpMotionId mid )
	:	hknpApiCommand( id, CMD_CREATE_BODY, sizeof(*this) )
	{
		m_motionId = mid;
	}

	hknpMotionId m_motionId;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpCreateBodyCommand, CMD_CREATE_BODY);


/// Destroy a body command.
struct hknpDestroyBodyCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpDestroyBodyCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpDestroyBodyCommand( hknpBodyId id )
	:	hknpApiCommand( id, CMD_DESTROY_BODY, sizeof(*this) ) {}
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpDestroyBodyCommand, CMD_DESTROY_BODY);


/// Add body command.
struct hknpAddBodyCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpAddBodyCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpAddBodyCommand( hknpBodyId id, hknpWorld::AdditionFlags additionFlags, bool isLastInBatch )
	:	hknpApiCommand( id, CMD_ADD_BODY, sizeof(*this) )
	{
		m_additionFlags = additionFlags;
		m_isLastInBatch = isLastInBatch;
	}

	hknpWorld::AdditionFlags m_additionFlags;
	hkBool m_isLastInBatch;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpAddBodyCommand, CMD_ADD_BODY);


/// Remove body command.
struct hknpRemoveBodyCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpRemoveBodyCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpRemoveBodyCommand( hknpBodyId id, bool isLastInBatch = false )
	:	hknpApiCommand( id, CMD_REMOVE_BODY, sizeof(*this) )
	{
		m_isLastInBatch = isLastInBatch;
	}

	hkBool m_isLastInBatch;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpRemoveBodyCommand, CMD_REMOVE_BODY);


/// Detach a body from a compound.
struct hknpDetachBodyCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpDetachBodyCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpDetachBodyCommand( hknpBodyId id, int updateInertia /*hknpWorld::UpdateInertia*/ )
	:	hknpApiCommand( id, CMD_DETACH_BODY, sizeof(*this))
	{
		m_updateInertia = updateInertia;
	}

	int m_updateInertia;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpDetachBodyCommand, CMD_DETACH_BODY);


/// Attach a body to a compound.
struct hknpAttachBodyCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpAttachBodyCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpAttachBodyCommand( hknpBodyId bodyId, hknpBodyId compoundBodyId, int updateInertia /*hknpWorld::UpdateInertia*/ )
	:	hknpApiCommand( bodyId, CMD_ATTACH_BODY, sizeof(*this) )
	{
		m_updateInertia = updateInertia;
		m_compoundBodyId = compoundBodyId;
	}

	hknpBodyId m_compoundBodyId;
	int m_updateInertia;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpAttachBodyCommand, CMD_ATTACH_BODY);


/// Sets the body transform.
struct hknpSetBodyTransformCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyTransformCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyTransformCommand(
		hknpBodyId id, const hkTransform& transform,
		hknpActivationBehavior::Enum activationBehavior = hknpActivationBehavior::ACTIVATE )
	:	hknpApiCommand( id, CMD_SET_BODY_TRANSFORM, sizeof(*this) )
	{
		m_activationBehavior = activationBehavior;
		m_transform = transform;
	}

	hknpActivationBehavior::Enum m_activationBehavior;
	hkTransform m_transform;
};
HK_DECLARE_COMMAND_DISCRIMINATOR( hknpSetBodyTransformCommand, CMD_SET_BODY_TRANSFORM );


/// Set the body position.
struct hknpSetBodyPositionCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyPositionCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyPositionCommand(
		hknpBodyId id, hkVector4Parameter position,
		hknpActivationBehavior::Enum activationBehavior = hknpActivationBehavior::ACTIVATE )
	:	hknpApiCommand( id, CMD_SET_BODY_POSITION, sizeof(*this) )
	{
		m_activationBehavior = activationBehavior;
		m_position = position;
	}

	hknpActivationBehavior::Enum m_activationBehavior;
	hkVector4 m_position;
};
HK_DECLARE_COMMAND_DISCRIMINATOR( hknpSetBodyPositionCommand, CMD_SET_BODY_POSITION );


/// Set the body orientation.
struct hknpSetBodyOrientationCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyOrientationCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyOrientationCommand(
		hknpBodyId id, hkQuaternionParameter orientation, hknpWorld::PivotLocation pivot,
		hknpActivationBehavior::Enum activationBehavior = hknpActivationBehavior::ACTIVATE )
		:	hknpApiCommand( id, CMD_SET_BODY_ORIENTATION, sizeof(*this) )
	{
		m_pivot					= pivot;
		m_activationBehavior	= activationBehavior;
		m_orientation			= orientation;
	}

	hknpWorld::PivotLocation		m_pivot;
	hknpActivationBehavior::Enum	m_activationBehavior;
	hkQuaternion					m_orientation;
};
HK_DECLARE_COMMAND_DISCRIMINATOR( hknpSetBodyOrientationCommand, CMD_SET_BODY_ORIENTATION );


/// Set the body velocity (both linear and angular).
struct hknpSetBodyVelocityCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyVelocityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyVelocityCommand( hknpBodyId id, hkVector4Parameter linVel, hkVector4Parameter angVel )
	:	hknpApiCommand( id, CMD_SET_BODY_VELOCITY, sizeof(*this) )
	{
		m_linVelocity = linVel;
		m_angVelocity = angVel;
	}

	hkVector4 m_linVelocity;
	hkVector4 m_angVelocity;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpSetBodyVelocityCommand, CMD_SET_BODY_VELOCITY);


/// Set a body's linear velocity.
struct hknpSetBodyLinearVelocityCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyLinearVelocityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyLinearVelocityCommand( hknpBodyId id, hkVector4Parameter linearVelocity )
	:	hknpApiCommand( id, CMD_SET_BODY_LINEAR_VELOCITY, sizeof(*this) )
	{
		m_linearVelocity = linearVelocity;
	}

	hkVector4 m_linearVelocity;
};
HK_DECLARE_COMMAND_DISCRIMINATOR( hknpSetBodyLinearVelocityCommand, CMD_SET_BODY_LINEAR_VELOCITY );


/// Set a body's angular velocity.
struct hknpSetBodyAngularVelocityCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyAngularVelocityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyAngularVelocityCommand( hknpBodyId id, hkVector4Parameter angularVelocity )
	:	hknpApiCommand( id, CMD_SET_BODY_ANGULAR_VELOCITY, sizeof(*this) )
	{
		m_angularVelocity = angularVelocity;
	}

	hkVector4 m_angularVelocity;
};
HK_DECLARE_COMMAND_DISCRIMINATOR( hknpSetBodyAngularVelocityCommand, CMD_SET_BODY_ANGULAR_VELOCITY );


/// Redo integration starting with last frame by a fraction of the time step.
struct hknpReintegrateBodyCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpReintegrateBodyCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpReintegrateBodyCommand( hknpBodyId id, hkReal t )
	:	hknpApiCommand( id, CMD_REINTEGRATE_BODY, sizeof(*this) )
	{
		m_t = t;
	}

	hkReal m_t;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpReintegrateBodyCommand, CMD_REINTEGRATE_BODY);


/// Apply linear impulse command.
struct hknpApplyLinearImpulseCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpApplyLinearImpulseCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpApplyLinearImpulseCommand( hknpBodyId id, hkVector4Parameter impulse )
	:	hknpApiCommand( id, CMD_APPLY_LINEAR_IMPULSE, sizeof(*this) )
	{
		m_impulse = impulse;
	}

	hkVector4 m_impulse;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpApplyLinearImpulseCommand, CMD_APPLY_LINEAR_IMPULSE);


/// Apply angular impulse command.
struct hknpApplyAngularImpulseCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpApplyAngularImpulseCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpApplyAngularImpulseCommand( hknpBodyId id, hkVector4Parameter impulse )
	:	hknpApiCommand( id, CMD_APPLY_ANGULAR_IMPULSE, sizeof(*this) )
	{
		m_impulse = impulse;
	}

	hkVector4 m_impulse;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpApplyAngularImpulseCommand, CMD_APPLY_ANGULAR_IMPULSE);


/// Apply point impulse command.
struct hknpApplyPointImpulseCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpApplyPointImpulseCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpApplyPointImpulseCommand( hknpBodyId id, hkVector4Parameter impulse, hkVector4Parameter positionWs )
	:	hknpApiCommand( id, CMD_APPLY_POINT_IMPULSE, sizeof(*this) )
	{
		m_impulse = impulse;
		m_position = positionWs;
	}

	hkVector4 m_impulse;
	hkVector4 m_position;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpApplyPointImpulseCommand, CMD_APPLY_POINT_IMPULSE);


/// Set body velocity at command.
struct hknpSetPointVelocityCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetPointVelocityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetPointVelocityCommand( hknpBodyId id, hkVector4Parameter velocity, hkVector4Parameter positionWs )
	:	hknpApiCommand( id, CMD_SET_POINT_VELOCITY, sizeof(*this) )
	{
		m_velocity = velocity;
		m_position = positionWs;
	}

	hkVector4 m_velocity;
	hkVector4 m_position;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpSetPointVelocityCommand, CMD_SET_POINT_VELOCITY);


/// Set body mass command.
struct hknpSetBodyMassCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyMassCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyMassCommand( hknpBodyId bodyId, hkReal massOrNegativeDensity, hknpWorld::RebuildCachesMode cacheBehavior )
	:	hknpApiCommand( bodyId, CMD_SET_BODY_MASS, sizeof(*this) )
	{
		m_massOrNegativeDensity = massOrNegativeDensity;
		m_cacheBehavior = hkUchar(cacheBehavior);
	}

	hkReal m_massOrNegativeDensity;
	hkUchar m_cacheBehavior; 
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpSetBodyMassCommand, CMD_SET_BODY_MASS);


/// Set body motion command.
struct hknpSetBodyMotionCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyMotionCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyMotionCommand( hknpBodyId bodyId, hknpMotionId motionId, hknpWorld::RebuildCachesMode cacheBehavior )
		:	hknpApiCommand( bodyId, CMD_SET_BODY_MOTION, sizeof(*this) )
	{
		m_motionId = motionId;
		m_cacheBehavior = hkUchar(cacheBehavior);
	}

	hknpMotionId m_motionId;
	hkUchar m_cacheBehavior; 
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpSetBodyMotionCommand, CMD_SET_BODY_MOTION);


/// Set the center of mass of a body.
struct hknpSetBodyCenterOfMassCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyCenterOfMassCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyCenterOfMassCommand( hknpBodyId id, hkVector4Parameter com )
	:	hknpApiCommand( id, CMD_SET_BODY_CENTER_OF_MASS, sizeof(*this) )
	{
		m_com = com;
	}

	hkVector4 m_com;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpSetBodyCenterOfMassCommand, CMD_SET_BODY_CENTER_OF_MASS);


/// Set the shape of the body.
struct hknpSetBodyShapeCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyShapeCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyShapeCommand( hknpBodyId id, const hknpShape* shape )
	:	hknpApiCommand( id, CMD_SET_BODY_SHAPE, sizeof(*this) )
	{
		m_shape = shape;
	}

	const hknpShape* m_shape;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpSetBodyShapeCommand, CMD_SET_BODY_SHAPE);


/// Set new motion properties to a body.
struct hknpSetBodyMotionPropertiesCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyMotionPropertiesCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyMotionPropertiesCommand( hknpBodyId id, hknpMotionPropertiesId mp )
	:	hknpApiCommand( id, CMD_SET_BODY_MOTION_PROPERTIES, sizeof(*this) )
	{
		m_motionProperties = mp;
	}

	hknpMotionPropertiesId m_motionProperties;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpSetBodyMotionPropertiesCommand, CMD_SET_BODY_MOTION_PROPERTIES);


/// Set a new materialId for a body.
struct hknpSetBodyMaterialCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyMaterialCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyMaterialCommand( hknpBodyId id, hknpMaterialId mp, int collisionCacheBehavior )
	:	hknpApiCommand( id, CMD_SET_BODY_MATERIAL, sizeof(*this) )
	{
		m_material = mp;
		m_collisionCacheBehavior = hkUchar(collisionCacheBehavior);
	}

	hknpMaterialId m_material;
	hkUchar m_collisionCacheBehavior;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpSetBodyMaterialCommand, CMD_SET_BODY_MATERIAL);


/// Set a new quality for a body.
struct hknpSetBodyQualityCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyQualityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyQualityCommand( hknpBodyId id, hknpBodyQualityId qualityId, int collisionCacheBehavior )
	:	hknpApiCommand( id, CMD_SET_BODY_QUALITY, sizeof(*this) )
	{
		m_qualityId = qualityId;
		m_collisionCacheBehavior = hkUchar(collisionCacheBehavior);
	}

	hknpBodyQualityId m_qualityId;
	hkUchar m_collisionCacheBehavior;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpSetBodyQualityCommand, CMD_SET_BODY_QUALITY);


/// Activate a body.
struct hknpActivateBodyCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpActivateBodyCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpActivateBodyCommand( hknpBodyId id, bool active )
	:	hknpApiCommand( id, CMD_ACTIVATE_BODY, sizeof(*this) )
	{
		m_active = active;
	}

	hkBool m_active;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpActivateBodyCommand, CMD_ACTIVATE_BODY);


/// Set a new collision filter on a body.
struct hknpSetBodyCollisionFilterInfoCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyCollisionFilterInfoCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyCollisionFilterInfoCommand( hknpBodyId id, hkUint32 collisionFilterInfo )
	:	hknpApiCommand( id, CMD_SET_BODY_FILTER_INFO, sizeof(*this) )
	{
		m_collisionFilterInfo = collisionFilterInfo;
	}

	hkUint32 m_collisionFilterInfo;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpSetBodyCollisionFilterInfoCommand, CMD_SET_BODY_FILTER_INFO);


/// Reset body collision caches command.
struct hknpRebuildBodyCollisionCachesCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpRebuildBodyCollisionCachesCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpRebuildBodyCollisionCachesCommand( hknpBodyId id )
	:	hknpApiCommand( id, CMD_REBUILD_BODY_COLLISION_CACHES, sizeof(*this) ) {}
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpRebuildBodyCollisionCachesCommand, CMD_REBUILD_BODY_COLLISION_CACHES);


/// Set the gravity on the world.
struct hknpSetWorldGravityCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetWorldGravityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetWorldGravityCommand( hkVector4Parameter gravity )
	:	hknpApiCommand( hknpBodyId(0), CMD_SET_WORLD_GRAVITY, sizeof(*this) )
	{
		m_gravity = gravity;
	}

	hkVector4 m_gravity;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpSetWorldGravityCommand, CMD_SET_WORLD_GRAVITY);


/// Set the material dynamic friction.
struct hknpSetMaterialFrictionCommand: public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetMaterialFrictionCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetMaterialFrictionCommand( hknpMaterialId materialId, hkReal dynamicFriction )
		:	hknpApiCommand( hknpBodyId(0), CMD_SET_MATERIAL_FRICTION, sizeof(*this)),
			m_materialId(materialId)
	{
		m_dynamicFriction.setReal<false>(dynamicFriction);
	}

	hknpMaterialId m_materialId;
	hkHalf m_dynamicFriction;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpSetMaterialFrictionCommand, CMD_SET_MATERIAL_FRICTION);


/// Set the material restitution.
struct hknpSetMaterialRestitutionCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetMaterialRestitutionCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetMaterialRestitutionCommand( hknpMaterialId materialId, hkReal restitution )
	:	hknpApiCommand( hknpBodyId(0), CMD_SET_MATERIAL_FRICTION, sizeof(*this) ),
		m_materialId(materialId)
	{
		m_restitution.setReal<false>(restitution);
	}

	hknpMaterialId m_materialId;
	hkHalf m_restitution;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpSetMaterialRestitutionCommand, CMD_SET_MATERIAL_RESTITUTION);


/// Sets the look ahead distance, see hknpBodyCinfo::m_collisionLookAheadDistance for details.
struct hknpSetBodyCollisionLookAheadDistanceCommand : public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpSetBodyCollisionLookAheadDistanceCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpSetBodyCollisionLookAheadDistanceCommand( hknpBodyId id, hkReal distance,
												  hkVector4Parameter tempExpansionVelocity )
	:	hknpApiCommand( hknpBodyId(id), CMD_SET_BODY_COLLISION_LOOKAHEAD_DISTANCE, sizeof(*this) )
	{
		m_distance = distance;
		m_tempExpansionVelocity = tempExpansionVelocity;
	}

	hkReal m_distance;
	hkVector4 m_tempExpansionVelocity;
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpSetBodyCollisionLookAheadDistanceCommand, CMD_SET_BODY_COLLISION_LOOKAHEAD_DISTANCE);


//
// Reserved commands
//

struct hknpReserved1VelocityCommand: public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpReserved1VelocityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpReserved1VelocityCommand( hknpBodyId id )
	:	hknpApiCommand( id, CMD_RESERVED_1, sizeof(*this) ) {}
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpReserved1VelocityCommand, CMD_RESERVED_1);

struct hknpReserved2VelocityCommand: public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpReserved2VelocityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpReserved2VelocityCommand( hknpBodyId id )
	:	hknpApiCommand( id, CMD_RESERVED_2, sizeof(*this) ) {}
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpReserved2VelocityCommand, CMD_RESERVED_2);

struct hknpReserved3VelocityCommand: public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpReserved3VelocityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpReserved3VelocityCommand( hknpBodyId id )
	:	hknpApiCommand( id, CMD_RESERVED_3, sizeof(*this) ) {}
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpReserved3VelocityCommand, CMD_RESERVED_3);

struct hknpReserved4VelocityCommand: public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpReserved4VelocityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpReserved4VelocityCommand( hknpBodyId id )
	:	hknpApiCommand( id, CMD_RESERVED_4, sizeof(*this) ) {}
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpReserved4VelocityCommand, CMD_RESERVED_4);

struct hknpReserved5VelocityCommand: public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpReserved5VelocityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpReserved5VelocityCommand( hknpBodyId id )
	:	hknpApiCommand( id, CMD_RESERVED_5, sizeof(*this) ) {}
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpReserved5VelocityCommand, CMD_RESERVED_5);

struct hknpReserved6VelocityCommand: public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpReserved6VelocityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpReserved6VelocityCommand( hknpBodyId id )
	:	hknpApiCommand( id, CMD_RESERVED_6, sizeof(*this) ) {}
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpReserved6VelocityCommand, CMD_RESERVED_6);

struct hknpReserved7VelocityCommand: public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpReserved7VelocityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpReserved7VelocityCommand( hknpBodyId id )
	:	hknpApiCommand( id, CMD_RESERVED_7, sizeof(*this) ) {}
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpReserved7VelocityCommand, CMD_RESERVED_7);

struct hknpReserved8VelocityCommand: public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpReserved8VelocityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpReserved8VelocityCommand( hknpBodyId id )
	:	hknpApiCommand( id, CMD_RESERVED_8, sizeof(*this) ) {}
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpReserved8VelocityCommand, CMD_RESERVED_8);

struct hknpReserved9VelocityCommand: public hknpApiCommand
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpReserved9VelocityCommand );
	HKNP_DECLARE_DEFAULT_COMMAND_FUNCTIONS;

	hknpReserved9VelocityCommand( hknpBodyId id )
	:	hknpApiCommand( id, CMD_RESERVED_9, sizeof(*this) ) {}
};
HK_DECLARE_COMMAND_DISCRIMINATOR(hknpReserved9VelocityCommand, CMD_RESERVED_9);


#endif	// HKNP_COMMANDS_H

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
