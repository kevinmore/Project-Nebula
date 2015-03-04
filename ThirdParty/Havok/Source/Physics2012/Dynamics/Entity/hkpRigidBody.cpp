/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#include <Physics2012/Dynamics/Motion/Rigid/hkpFixedRigidMotion.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpKeyframedRigidMotion.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpCharacterMotion.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpSphereMotion.h>
#include <Physics2012/Dynamics/Motion/Rigid/ThinBoxMotion/hkpThinBoxMotion.h>

#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Physics2012/Collide/Shape/hkpShape.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>

#include <Physics2012/Dynamics/Destruction/BreakableBody/hkpBreakableBody.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationQueue.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldAgentUtil.h>

#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>
#include <Physics2012/Dynamics/World/Simulation/hkpSimulation.h>

#include <Physics2012/Dynamics/Entity/Util/hkpEntityCallbackUtil.h>

#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>
#include <Common/Base/Types/Geometry/LocalFrame/hkLocalFrame.h>
#include <Physics2012/Collide/Util/ShapeDepth/hkpShapeDepthUtil.h>

HK_COMPILE_TIME_ASSERT( sizeof(hkpRigidBody) == sizeof(hkpEntity) );
#if HK_POINTER_SIZE == 4 && !defined(HK_PLATFORM_PS3_SPU) && !defined(HK_REAL_IS_DOUBLE)
	#include <Common/Base/Memory/Allocator/Thread/hkThreadMemory.h>
	HK_COMPILE_TIME_ASSERT( sizeof(hkpRigidBody) <= hkThreadMemory::MEMORY_MAX_SIZE_SMALL_BLOCK );
#endif

static void estimateAllowedPenetrationDepth(hkpCollidable* collidable, const hkReal allowedPenetrationDepth, const hkVector4& extent)
{
	const hkSimdReal minObjectDiameter = extent.horizontalMin<3>();

	HK_ASSERT2(0xad65bbe4, minObjectDiameter.isGreaterZero(), "Minimum object diameter cannot equal zero. The problem may be caused by the convex hull generator. E.g. in case of objects smaller than 1cm in size, all vertices may get welded together.");

	if ( minObjectDiameter.isLess(hkSimdReal_Inv2) )
	{
		collidable->m_allowedPenetrationDepth = (minObjectDiameter * hkSimdReal_Inv5).getReal(); // = (1.0f / radiusToAllowedPenetrationRatio)
	}
	else
	{
		collidable->m_allowedPenetrationDepth = hkReal(0.1f); 
	}

	HK_ASSERT2(0xad65bbe4, collidable->m_allowedPenetrationDepth > 0.00001f, "Allowed penetration depth cannot be zero. (This assert fires when it is less than 0.00001 (0.01mm), so if you know what you're doing you may ignore it.)");
	if (collidable->m_allowedPenetrationDepth < 0.001f)
	{
		HK_WARN(0xad65bbe4, "Allowed penetration depth cannot be zero. (This warning fires when it is less than 0.001 (1mm), so if you know what you're doing you may ignore it.)");
	}
}

void hkpRigidBody::updateCachedShapeInfo(const hkpShape* shape, hkVector4& extentOut )
{
	HK_ASSERT2(0x7cdcd3a0, shape, "The rigid body must have a shape.");
	// get the radius
	hkAabb aabb;
	shape->getAabb( hkTransform::getIdentity(), 0.0f, aabb );
	hkCheckDeterminismUtil::checkMt(0xf0000211, aabb);

	extentOut.setSub( aabb.m_max, aabb.m_min );

	hkVector4 com = getCenterOfMassLocal();
	hkVector4 aabbMassRelativeMax; aabbMassRelativeMax.setSub( aabb.m_max, com );
	hkVector4 aabbMassRelativeMin; aabbMassRelativeMin.setSub( aabb.m_min, com );

	const hkSimdReal radius0 = aabbMassRelativeMax.lengthSquared<3>();
	const hkSimdReal radius1 = aabbMassRelativeMin.lengthSquared<3>();

	hkMotionState& ms = getRigidMotion()->m_motionState;
	hkSimdReal maxR; maxR.setMax(radius0,radius1);
	ms.m_objectRadius = maxR.sqrt().getReal();
}




void HK_CALL hkpRigidBody::createDynamicRigidMotion( hkpMotion::MotionType motionType, const hkVector4& position, const hkQuaternion& rotation, 
													hkReal mass, const hkMatrix3& inertiaLocal, const hkVector4& centreOfMassLocal, hkReal maxLinearVelocity, hkReal maxAngularVelocity,
													hkpMaxSizeMotion* motionBufferOut )
{

	hkCheckDeterminismUtil::checkMt(0xf0000020, motionType);
	hkCheckDeterminismUtil::checkMt(0xf0000021, position);
	hkCheckDeterminismUtil::checkMt(0xf0000022, rotation);
	hkCheckDeterminismUtil::checkMt(0xf0000023, mass);
	hkCheckDeterminismUtil::checkMt(0xf0000024, inertiaLocal);
	hkCheckDeterminismUtil::checkMt(0xf0000025, centreOfMassLocal);
	hkCheckDeterminismUtil::checkMt(0xf0000026, maxLinearVelocity);
	hkCheckDeterminismUtil::checkMt(0xf0000027, maxAngularVelocity);


	if(motionType != hkpMotion::MOTION_KEYFRAMED)
	{
		HK_ASSERT2(0x305ac731,  mass > 0, "Mass not valid (mass > 0)" );
	}

	hkpMotion* motion;
	switch(motionType)
	{
		case hkpMotion::MOTION_SPHERE_INERTIA:
		{
			motion = new (motionBufferOut) hkpSphereMotion(position, rotation );
			break;
		}
		
		case hkpMotion::MOTION_BOX_INERTIA:
		{
			motion = new (motionBufferOut) hkpBoxMotion(position, rotation );
			break;
		}
					
		case hkpMotion::MOTION_THIN_BOX_INERTIA:
		{
			if ( maxAngularVelocity > 10.0f )
			{
				HK_WARN( 0xf03243df, "To get a stable thin box working, it is adviced to limit the angular velocity of the rigid body to be ~ 4.0f rad/sec" );
			}
			motion = new (motionBufferOut) hkpThinBoxMotion(position, rotation );
			break;
		}
		case hkpMotion::MOTION_DYNAMIC:
		{
			hkVector4 diag; hkMatrix3Util::_getDiagonal(inertiaLocal, diag);

			const hkSimdReal ma = diag.horizontalMax<3>();
			const hkSimdReal mi = diag.horizontalMin<3>();

			if ( mi > ma * hkSimdReal::fromFloat(0.8f) )
			{
				motion = new (motionBufferOut) hkpSphereMotion(position, rotation );
			}
			else
			{
				motion = new (motionBufferOut) hkpBoxMotion(position, rotation );
			}
			break;
		}
		
		case hkpMotion::MOTION_KEYFRAMED:
		{
			maxLinearVelocity    = 1e6f;
			maxAngularVelocity	 = 1e6f;
			motion = new (motionBufferOut) hkpKeyframedRigidMotion(position, rotation );
			break;
		}

		case hkpMotion::MOTION_CHARACTER:
		{
			motion = new (motionBufferOut) hkpCharacterMotion(position, rotation );
			break;
		}
		
		default:
		{
			motion = new (motionBufferOut) hkpFixedRigidMotion( position, rotation );
			HK_ASSERT2(0x6fd67199, 0,"Motiontype invalid in RigidBody constuction info. Cannot construct body.");
		}
	}

	// If not a keyframed body, set the mass properties...
	if(motionType != hkpMotion::MOTION_KEYFRAMED)
	{
		// Default value initialised to Diag(-1,-1,-1). Confirm that user has overwritten this
		// if we're about to use it for mass properties
		HK_ASSERT2(0x11a9ad41, (inertiaLocal(0,0) > 0.f)	|| (inertiaLocal(1,1) > 0.f)	|| (inertiaLocal(2,2) > 0.f), 
					"You have to specify a valid inertia tensor for rigid bodies in the hkpRigidBodyCinfo.");	

		// Default value initialised to -1. Confirm that user has overwritten this
		// if we're about to use it for mass properties
		HK_ASSERT2(0x245a804f,  mass > 0, "You have to specify a valid mass for rigid bodies in the hkpRigidBodyCinfo.");	

		motion->setInertiaLocal(inertiaLocal);
		motion->setCenterOfMassInLocal( centreOfMassLocal ); 
		motion->setMass(mass);
	}

	motion->getMotionState()->m_maxLinearVelocity = hkFloat32(maxLinearVelocity);
	motion->getMotionState()->m_maxAngularVelocity = hkFloat32(maxAngularVelocity);
	motion->m_savedQualityTypeIndex = 0;
	// Info: this doesn't set motion->getMotionState()->m_objectRadius. 
	// This function is only meant to be used in hkpRigidBody constructor and in void hkpWorldOperationUtil::replaceMotionObject()
	// where it is followed by a call to either setMotionDeltaAngleMultiplier() or hkpMotion::getPositionAndVelocities( newMotion );
}

hkpRigidBody::hkpRigidBody( const hkpRigidBodyCinfo& info )
:	hkpEntity( info.m_shape )
{
	//
	// Set hkpEntity and hkpWorld object properties
	//
	m_material.setResponseType( info.m_collisionResponse );
	HK_WARN_ON_DEBUG_IF( info.m_collisionResponse == hkpMaterial::RESPONSE_REPORTING, 0x23a78ac2, "hkpMaterial::RESPONSE_REPORTING is deprecated. Instead, you can use the disableConstraint modifier." );
	m_contactPointCallbackDelay = info.m_contactPointCallbackDelay;
	m_collidable.setCollisionFilterInfo( info.m_collisionFilterInfo );

	// set hkpEntity::m_localFrame
	m_localFrame = info.m_localFrame;

	HK_ASSERT2(0x492fb4e0,
		info.m_motionType > hkpMotion::MOTION_INVALID && info.m_motionType < hkpMotion::MOTION_MAX_ID,
		"Motiontype invalid (not yet specified) in RigidBody constuction info. Cannot construct body.");

	HK_ASSERT2( 0xf0010434, info.m_solverDeactivation > hkpRigidBodyCinfo::SOLVER_DEACTIVATION_INVALID, "m_solverDeactivation not set");
	
	if (info.m_motionType == hkpMotion::MOTION_FIXED)
	{
		hkpFixedRigidMotion* p = new (&m_motion) hkpFixedRigidMotion(info.m_position, info.m_rotation);

		p->getMotionState()->m_maxLinearVelocity  = hkFloat32(info.m_maxLinearVelocity);
		p->getMotionState()->m_maxAngularVelocity = hkFloat32(info.m_maxAngularVelocity);

		p->setDeactivationClass( hkpRigidBodyCinfo::SOLVER_DEACTIVATION_OFF );

		hkMotionState* ms = getRigidMotion()->getMotionState();
		m_collidable.setMotionState( ms);

		if ( info.m_allowedPenetrationDepth <= 0.0f )
		{
			m_collidable.m_allowedPenetrationDepth = HK_REAL_MAX;
		}
		else
		{
			m_collidable.m_allowedPenetrationDepth = info.m_allowedPenetrationDepth;
		}
	}
	else
	{
		hkpRigidBody::createDynamicRigidMotion( info.m_motionType, info.m_position, info.m_rotation, info.m_mass, info.m_inertiaTensor, info.m_centerOfMass, info.m_maxLinearVelocity, info.m_maxAngularVelocity, &m_motion );
		hkpMotion* rigidMotion  = &m_motion;
		rigidMotion->setDeactivationClass( info.m_solverDeactivation );

		setLinearVelocity(info.m_linearVelocity);
		setAngularVelocity(info.m_angularVelocity);

		m_collidable.setMotionState( getRigidMotion()->getMotionState()); 

		// User is doing 'lazy' construction.
		getCollidableRw()->m_allowedPenetrationDepth = info.m_allowedPenetrationDepth;
	}

	// Set deactivation state for this rigid body		
	enableDeactivation(info.m_enableDeactivation);

	setLinearDamping( info.m_linearDamping );
	setAngularDamping( info.m_angularDamping );
	setTimeFactor( info.m_timeFactor );

	const hkpShape* shape = getCollidable()->getShape();

	if( shape )
	{			
		hkVector4 extent;
		updateCachedShapeInfo( shape, extent);
		if ( getCollidable()->m_allowedPenetrationDepth <= 0.0f)
		{
			estimateAllowedPenetrationDepth(getCollidableRw(), getCollidable()->m_allowedPenetrationDepth, extent);
		}
#ifdef HK_DEBUG
		checkPerformance();
#endif
	}
	else
	{
		// only fixed body in the world has no shape
		HK_ASSERT2(0x7cdcd39f, info.m_motionType == hkpMotion::MOTION_FIXED, "Cannot create an entity without a shape.");
	}
	
	if ( info.m_qualityType != HK_COLLIDABLE_QUALITY_INVALID )
	{
		setQualityType( info.m_qualityType );
		HK_ASSERT2( 0xf0ed54f7, !isFixed() || (HK_COLLIDABLE_QUALITY_FIXED == info.m_qualityType), "Fixed objects must have the HK_COLLIDABLE_QUALITY_FIXED type" );
	}
	else
	{
		if ( isFixed() )
		{
			setQualityType( HK_COLLIDABLE_QUALITY_FIXED );
		}
		else if ( info.m_motionType == hkpMotion::MOTION_KEYFRAMED)
		{
			setQualityType( HK_COLLIDABLE_QUALITY_KEYFRAMED );
		}
		else
		{
			setQualityType( HK_COLLIDABLE_QUALITY_DEBRIS );
		}
	}

	m_autoRemoveLevel = info.m_autoRemoveLevel;
	if ( info.m_forceCollideOntoPpu )
	{
		m_collidable.m_forceCollideOntoPpu |= hkpCollidable::FORCE_PPU_USER_REQUEST;
	}

	getMaterial().setFriction( info.m_friction );
	getMaterial().setRollingFrictionMultiplier( info.m_rollingFrictionMultiplier );
	getMaterial().setRestitution( info.m_restitution );

	m_motion.m_gravityFactor.setReal<true>(info.m_gravityFactor);

	// If the provided value is negative, use the shapeDepthUtil to find out how deep the shape hierarchy is.
	if ( info.m_numShapeKeysInContactPointProperties >= 0 )
	{
		m_numShapeKeysInContactPointProperties = info.m_numShapeKeysInContactPointProperties;
#if defined(HK_DEBUG)
		// This warning can safely be ignored. Sufficient shape key storage however is required for Havok Destruction.
		if( ( 0 < m_numShapeKeysInContactPointProperties ) && ( m_numShapeKeysInContactPointProperties < hkpShapeDepthUtil::getShapeDepth( info.m_shape ) ) )
		{
			HK_WARN_ALWAYS(0x458a1bb2, "Shape key storage insufficient to store all possible branches. This may cause truncation of the shape key information." );
		}
#endif
	}
	else
	{
		m_numShapeKeysInContactPointProperties = hkpShapeDepthUtil::getShapeDepth( info.m_shape );
	}
	m_responseModifierFlags = info.m_responseModifierFlags;
}

hkpRigidBody::hkpRigidBody( class hkFinishLoadedObjectFlag flag ) : hkpEntity( flag )
{

}

hkpRigidBody::~hkpRigidBody()
{
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
}

void hkpRigidBody::getCinfo(hkpRigidBodyCinfo& info) const
{
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RO );

	// Entity attributes
	{
		info.m_forceCollideOntoPpu = (m_collidable.m_forceCollideOntoPpu & hkpCollidable::FORCE_PPU_USER_REQUEST) != 0;
		info.m_autoRemoveLevel = m_autoRemoveLevel;
		info.m_numShapeKeysInContactPointProperties = m_numShapeKeysInContactPointProperties;
		info.m_contactPointCallbackDelay = m_contactPointCallbackDelay;

		info.m_enableDeactivation = isDeactivationEnabled();

		info.m_localFrame = m_localFrame;

		// Material attributes
		{
			info.m_friction = getMaterial().getFriction();	
			info.m_rollingFrictionMultiplier = getMaterial().getRollingFrictionMultiplier();
			info.m_collisionResponse = m_material.getResponseType();	
			info.m_restitution = getMaterial().getRestitution();
		}

		// Motion attributes
		{
			info.m_linearDamping = getLinearDamping();		
			info.m_angularDamping = getAngularDamping();

			info.m_linearVelocity = getLinearVelocity();
			info.m_angularVelocity = getAngularVelocity();
	
			info.m_mass = getMass();			
			getInertiaLocal( info.m_inertiaTensor );

			info.m_motionType = getMotionType();

			// Motion state attributes
			{
				
				info.m_solverDeactivation =  static_cast<hkpRigidBodyCinfo::SolverDeactivation>(getRigidMotion()->getDeactivationClass());		

				info.m_maxLinearVelocity  = getMaxLinearVelocity();
				info.m_maxAngularVelocity = getMaxAngularVelocity();

				info.m_position = getPosition();
				info.m_rotation = getRotation();
				
				info.m_centerOfMass = getCenterOfMassLocal();
			}
		}	
	}

	// World Object attributes
	{
		// Collidable attributes
		{
			info.m_shape = m_collidable.getShape();
			info.m_collisionFilterInfo = m_collidable.getCollisionFilterInfo();
			info.m_allowedPenetrationDepth = getCollidable()->m_allowedPenetrationDepth;
			info.m_qualityType = getCollidable()->getQualityType();	
		}
	}		

	info.m_gravityFactor = m_motion.m_gravityFactor;
}

hkMotionState* hkpRigidBody::getMotionState()
{
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_RW, this, HK_ACCESS_RW );
	return getRigidMotion()->getMotionState();
}

hkpRigidBody* hkpRigidBody::clone() const 
{
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RO );

	// Gather static shape info as well as dynamics motion state info
    hkpRigidBodyCinfo currentData;
	{
		getCinfo( currentData );
	}

	// Initialize clone
	hkpRigidBody* rbClone = new hkpRigidBody( currentData );

	// Overwrite hkpMotion of the body with a true clone
	{
			// we need memcpy, so that the vtable pointer is copied as well
		hkString::memCpy16NonEmpty( &rbClone->m_motion, &m_motion, sizeof( hkpMaxSizeMotion)>>4);
		if ( m_motion.m_savedMotion )
		{
			rbClone->m_motion.m_savedMotion = new hkpMaxSizeMotion();

			// Preserve the m_memSizeAndFlags and m_referenceCount which will
			// be overwritten by the memcpy below
			hkUint16 oldSMMemSizeAndFlags = rbClone->m_motion.m_savedMotion->m_memSizeAndFlags;
			hkUint16 oldSMReferenceCount = rbClone->m_motion.m_savedMotion->m_referenceCount;

			hkString::memCpy16NonEmpty( rbClone->m_motion.m_savedMotion, m_motion.m_savedMotion, sizeof( hkpMaxSizeMotion)>>4);

			rbClone->m_motion.m_savedMotion->m_memSizeAndFlags = oldSMMemSizeAndFlags;
			rbClone->m_motion.m_savedMotion->m_referenceCount = oldSMReferenceCount;
		}
		// The CdBody has ptr back to the motion too, so we need to fix that up
		rbClone->getCollidableRw()->setMotionState( rbClone->getMotionState());
	}

	// Add user information to clone (properties)
	rbClone->copyProperties( this );
	{
		rbClone->setName( getName() );
		rbClone->setUserData( getUserData() );	
	}

	if ( m_breakableBody )
	{
		rbClone->m_breakableBody = m_breakableBody->cloneBreakableBody(rbClone);
	}

	return rbClone;
}


void hkpRigidBody::setMotionType( hkpMotion::MotionType newState, hkpEntityActivation preferredActivationState, hkpUpdateCollisionFilterOnEntityMode collisionFilterUpdateMode )
{ 
	hkCheckDeterminismUtil::checkMt(0xf0000212, newState);

	if (getWorld() && getWorld()->areCriticalOperationsLocked())
	{
		hkWorldOperation::SetRigidBodyMotionType op;
		op.m_rigidBody = this;
		op.m_motionType = newState;
		op.m_activation = preferredActivationState;
		op.m_collisionFilterUpdateMode = collisionFilterUpdateMode;

		getWorld()->queueOperation(op);
		return;
	}

#ifdef HK_DEBUG
	// we must keep an extra reference for the body if we want to run checkPerformace() below.
	// hkpWorldOperationUtil::setRigidBodyMotionType() triggers callbacks, which may remove the body from the world and destroy it. 
	this->addReference();
#endif

	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_RW, this, HK_ACCESS_RW );
	hkpWorldOperationUtil::setRigidBodyMotionType(this, newState, preferredActivationState, collisionFilterUpdateMode);

	if (m_world)
	{
		hkpWorldCallbackUtil::fireEntitySetMotionType( m_world, this );
	}
	hkpEntityCallbackUtil::fireEntitySetMotionType( this );

#ifdef HK_DEBUG
	checkPerformance();
	this->removeReference();
#endif
}


hkWorldOperation::Result hkpRigidBody::setShape(const hkpShape* shape )
{
	HK_ASSERT2(0x2005c7ff, shape, "Cannot setShape to NULL.");

	if (m_world && m_world->areCriticalOperationsLocked())
	{
		hkWorldOperation::SetWorldObjectShape op;
		op.m_worldObject = this;
		op.m_shape = shape;

		m_world->queueOperation(op);
		return hkWorldOperation::POSTPONED;
	}

	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_RW, this, HK_ACCESS_RW );

	if (m_world)
	{
		m_world->lockCriticalOperations();

		hkpWorldOperationUtil::removeEntityBP(m_world, this);
	}


	{
		const hkpShape* previousShape = getCollidable()->getShape();

		// Handle reference counting here.
		getCollidableRw()->setShape(shape);
		shape->addReference();

		if (previousShape)
		{
			previousShape->removeReference();
		}

#ifdef HK_DEBUG
		checkPerformance();
#endif

		// Perform additional necessary computations
		hkVector4 extent;
		updateCachedShapeInfo(shape, extent);

		if (previousShape && getCollidable()->m_allowedPenetrationDepth != HK_REAL_MAX)
		{
			// Force automatic recalculation of allowed penetration depth if a shape was already present in that collidable
			getCollidableRw()->m_allowedPenetrationDepth = -1.0f; 
		}
		if ( getCollidableRw()->m_allowedPenetrationDepth <= 0.0f)
		{
			estimateAllowedPenetrationDepth(getCollidableRw(), getCollidable()->m_allowedPenetrationDepth, extent);
		}
	}

	if ( m_world )
	{
		setCachedShapeData(m_world, shape); // THIS MUST BE DONE AFTER THE SHAPE IS ASSIGNED, BECAUSE IT USES THE SHAPE FROM THE COLLIDABLE !!!
	}


	// Callbacks -- done after all the other radii, allowedPenetrationDepths are recalculated.
	// and before the body is added back to broadphase
	if (m_world)
	{
		// Moreover it's better do it after 
		hkpWorldCallbackUtil::fireEntityShapeSet( m_world, this );
	}
	hkpEntityCallbackUtil::fireEntityShapeSet( this );


	if (m_world)
	{
		hkpWorldOperationUtil::addEntityBP(m_world, this);

		m_world->unlockAndAttemptToExecutePendingOperations();
	}

	return hkWorldOperation::DONE;
}

hkWorldOperation::Result hkpRigidBody::updateShape(hkpShapeModifier* shapeModifier)
{
	// Postpone operation if necessary
	if (m_world && m_world->areCriticalOperationsLocked())
	{
		hkWorldOperation::UpdateWorldObjectShape op;
		op.m_worldObject	= this;
		op.m_shapeModifier	= shapeModifier;

		m_world->queueOperation(op);
		return hkWorldOperation::POSTPONED;
	}

	// Lock world
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_RW, this, HK_ACCESS_RW );
	if (m_world)
	{
		m_world->lockCriticalOperations();
	}

	// Get shape
	const hkpShape* shape = getCollidable()->getShape();

	// Execute modifier if case
	if ( shapeModifier )
	{
		hkpShape* nonConstShape = const_cast<hkpShape*>(shape);
		shapeModifier->modifyShape(nonConstShape);
	}

	// Update shape cache
	{
		hkVector4 extent;
		updateCachedShapeInfo( shape, extent );
	}

	if( m_world )
	{
		// Invalidate AABB cache
		m_collidable.m_boundingVolumeData.invalidate();

		// The broad phase representations of fixed objects do not get updated during the step,
		// so we force an update now. This can add or remove collision agents.
		if ( isFixed() )
		{
			hkpEntity* e = this;
			m_world->m_simulation->resetCollisionInformationForEntities(&e, 1, m_world);
			hkpWorldOperationUtil::updateEntityBP( m_world, this );
		}

		// Unlock world
		m_world->unlockAndAttemptToExecutePendingOperations();
		
		hkpWorldCallbackUtil::fireEntityShapeSet(m_world, this);
	}

	// Fire entity shape set event
	hkpEntityCallbackUtil::fireEntityShapeSet(this);

	return hkWorldOperation::DONE;
}

	// Explicit center of mass in local space
void hkpRigidBody::setCenterOfMassLocal(const hkVector4& centerOfMass)
{
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	hkCheckDeterminismUtil::checkMt(0xf0000213, centerOfMass);

	getRigidMotion()->setCenterOfMassInLocal(centerOfMass);
	hkVector4 dummyExtent;
	updateCachedShapeInfo(m_collidable.getShape(), dummyExtent);
}

void hkpRigidBody::enableDeactivation( bool _enableDeactivation )
{
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_RO, this, HK_ACCESS_RW );
	hkCheckDeterminismUtil::checkMt(0xf0000214, _enableDeactivation);	

	if (_enableDeactivation)
	{
		// Do not enable deactivation if it is already enabled to avoid reseting the deactivation counter
		if (isDeactivationEnabled())
		{
			return;
		}

		// Use spatial deactivation scheme.
		if (m_world) 	 
		{ 	 
			hkUint8* deactFlags = m_world->m_dynamicsStepInfo.m_solverInfo.m_deactivationNumInactiveFramesSelectFlag; 	 
			m_motion.enableDeactivation(true, m_uid, deactFlags[0], deactFlags[1], m_world->m_dynamicsStepInfo.m_solverInfo.m_deactivationIntegrateCounter);
		}
		else
		{
			m_motion.enableDeactivation(true, m_uid);
		}
	}
	else
	{
		m_motion.enableDeactivation(false);
	}
}

bool hkpRigidBody::isDeactivationEnabled() const
{
	return m_motion.isDeactivationEnabled();
}

hkpMotion* hkpRigidBody::getStoredDynamicMotion()
{
	if ( m_motion.getType() == hkpMotion::MOTION_KEYFRAMED || m_motion.getType() == hkpMotion::MOTION_FIXED )
	{
		return static_cast<hkpKeyframedRigidMotion*>( &m_motion )->m_savedMotion;
	}
	else
	{
		return HK_NULL;
	}
}


const hkpMotion* hkpRigidBody::getStoredDynamicMotion() const
{
	if ( m_motion.getType() == hkpMotion::MOTION_KEYFRAMED || m_motion.getType() == hkpMotion::MOTION_FIXED )
	{
		return static_cast<const hkpKeyframedRigidMotion*>( &m_motion )->m_savedMotion;
	}
	else
	{
		return HK_NULL;
	}
}



/*
** UTILITIY
*/

//#define USE_OPERATION_DELAY_MGR



void HK_CALL hkpRigidBody::updateBroadphaseAndResetCollisionInformationOfWarpedBody( hkpEntity* entity )
{
	hkCheckDeterminismUtil::checkMt(0xf0000215, entity->getUid() );
	hkpWorld* world = entity->getWorld();
	if (world)
	{
		if ( world->areCriticalOperationsLocked() )
		{
			hkWorldOperation::UpdateMovedBodyInfo op;
			op.m_entity = entity;
			world->queueOperation(op);
			return;
		}

		HK_ACCESS_CHECK_WITH_PARENT( world, HK_ACCESS_RW, entity, HK_ACCESS_RW );

		world->lockCriticalOperations();

		hkpEntity* entities[1] = { entity };
		// Note: it is intended to invalidate manifolds of the agents in this function.
		// As we have no interface to do it, for now, we call update broadphase, so that
		// agents, and therefore contact points, are removed when a body is moved over a large distance.
		// (No midphase is done though, so that doesn't work, when you're overlapping with a MOPP.)

		// TOI events removed in resetCollisionInformationForEntities
		world->m_simulation->resetCollisionInformationForEntities( entities, 1, world );

		if(entity->getCollidable()->getShape() != HK_NULL)
		{
			hkpSimulation::collideEntitiesBroadPhaseDiscrete(entities, 1, world);
		}

			// fixed bodies are also inactive, so we simplify the following check
		if (!entity->isActive())
		{
			if (world->m_shouldActivateOnRigidBodyTransformChange && !entity->isFixed())
			{
#if defined(HK_DEBUG)
				hkpSimulationIsland* island = entity->getSimulationIsland();
				if ( island->m_activeMark == false && island->m_isInActiveIslandsArray )
				{
					// a previous deactivation request is overwritten, try to make island sparse
					HK_WARN( 0xf034546f, "You are calling hkpRigidBody::setTransform of a RigidBody, which tries to deactivate. This keeps the RigidBody alive at some extra performance cost. Try to avoid calling unnecessary setTransform() or disable deactivation for this body");
					island->m_tryToIncreaseIslandSizeMark = true;
				}
#endif
				entity->activate();
			}
			else
			{
				// We want to update AABBs so that inactive bodies don't get waken up without no reason after 
				// This may fire immediate contactPointRemoved callbacks 

				// Temporarily BroadPhase Collision Detection executed for active bodies too (see above).
				// hkpSimulation::collideEntitiesBroadPhaseDiscrete(entities, 1, world);
			}

			// update graphics, or sth.
			hkpWorldCallbackUtil::fireInactiveEntityMoved(world, entity);
		}	

		world->unlockAndAttemptToExecutePendingOperations();
	}
}


	// Set the position (Local Space origin) of this rigid body, in World space.
void hkpRigidBody::setPosition(const hkVector4& position)
{
		// if you hit the next assert, consider using setPositionAndRotationAsCriticalOperation
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	hkCheckDeterminismUtil::checkMt(0xf0000216, position);
	HK_ASSERT2(0x43249f20, position.isOk<3>(), "Position passed to hkpRigidBody::setPosition is invalid");
	getRigidMotion()->setPosition(position);
	updateBroadphaseAndResetCollisionInformationOfWarpedBody(this);
}

	// Set the rotation from Local to World Space for this rigid body.
void hkpRigidBody::setRotation(const hkQuaternion& rotation)
{
		// if you hit the next assert, consider using setPositionAndRotationAsCriticalOperation
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	hkCheckDeterminismUtil::checkMt(0xf0000216, rotation);

	HK_ASSERT2(0x275ec1fd, rotation.isOk(), "hkQuaternion used in hkpRigidBody::setRotation() is not normalized/invalid!");

	getRigidMotion()->setRotation(rotation);
	updateBroadphaseAndResetCollisionInformationOfWarpedBody(this);
}

	// Set the position and rotation of the rigid body, in World space.
void hkpRigidBody::setPositionAndRotation(const hkVector4& position, const hkQuaternion& rotation)
{
		// if you hit the next assert, consider using setPositionAndRotationAsCriticalOperation
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	hkCheckDeterminismUtil::checkMt(0xf0000217, position);
	hkCheckDeterminismUtil::checkMt(0xf0000218, rotation);

	HK_ASSERT2(0x43249f20, position.isOk<3>(), "Position passed to hkpRigidBody::setPositionAndRotation is invalid");
	HK_ASSERT2(0x4669e4a0, rotation.isOk(), "hkQuaternion used in hkpRigidBody::setPositionAndRotation() is not normalized/invalid!");
	
	getRigidMotion()->setPositionAndRotation(position, rotation);
	updateBroadphaseAndResetCollisionInformationOfWarpedBody(this);
}

	// Sets the rigid body (local) to world transformation
void hkpRigidBody::setTransform(const hkTransform& transform)
{
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	hkCheckDeterminismUtil::checkMt(0xf0000220, transform);
	HK_ASSERT2(0x65fd5e95, transform.isOk(), "Transform passed to hkpRigidBody::setTransform is invalid");
	getRigidMotion()->setTransform(transform);
	updateBroadphaseAndResetCollisionInformationOfWarpedBody(this);
}

	// Set the mass of the rigid body.
void hkpRigidBody::setMass(hkReal m)
{
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	hkCheckDeterminismUtil::checkMt(0xf0000221, m);

	getRigidMotion()->setMass(m);
}

void hkpRigidBody::setMassInv(hkReal mInv)
{
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	hkCheckDeterminismUtil::checkMt(0xf0000222, mInv);

	getRigidMotion()->setMassInv(mInv);
}

void hkpRigidBody::setFriction( hkReal newFriction )
{ 
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	hkCheckDeterminismUtil::checkMt(0xf0000223, newFriction);

	m_material.setFriction(newFriction); 
}

void hkpRigidBody::setRollingFrictionMultiplier(hkReal multiplier)
	{
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	hkCheckDeterminismUtil::checkMt(0xf0000224, multiplier);

	HK_ON_DEBUG(hkReal oldMultiplier = m_material.getRollingFrictionMultiplier());
	HK_ASSERT2(0xad811071, !getWorld() || oldMultiplier * multiplier != 0.0f || oldMultiplier + multiplier == 0.0f, 
			"Changing rolling friction between zero and non-zero values while the body is in the world is not allowed and may cause a crash" \
		"in the solver (due to rolling friction schemas not being accounted for when allocating the solver buffer). "\
		"Initialize the multiplier with a small value instead of zero.");
	m_material.setRollingFrictionMultiplier( multiplier );
}

void hkpRigidBody::setRestitution( hkReal newRestitution ) 
{ 
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	hkCheckDeterminismUtil::checkMt(0xf0000224, newRestitution);

	m_material.setRestitution(newRestitution); 
}

	// Sets the inertia tensor of the rigid body. Advanced use only.
void hkpRigidBody::setInertiaLocal(const hkMatrix3& inertia)
{
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	hkCheckDeterminismUtil::checkMt(0xf0000225, inertia);

	getRigidMotion()->setInertiaLocal(inertia);
}


	// Sets the inertia tensor of the rigid body by supplying its inverse. Advanced use only.
void hkpRigidBody::setInertiaInvLocal(const hkMatrix3& inertiaInv)
{
	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	hkCheckDeterminismUtil::checkMt(0xf0000227, inertiaInv);
	getRigidMotion()->setInertiaInvLocal(inertiaInv);
}

void hkpRigidBody::setPositionAndRotationAsCriticalOperation(const hkVector4& position, const hkQuaternion& rotation)
{
	hkCheckDeterminismUtil::checkMt(0xf0000228, position);
	hkCheckDeterminismUtil::checkMt(0xf0000229, rotation);

	if (getWorld() && getWorld()->areCriticalOperationsLocked())
	{
		hkWorldOperation::SetRigidBodyPositionAndRotation op;
		op.m_rigidBody = this;
		op.m_positionAndRotation = hkAllocateChunk<hkVector4>( 2, HK_MEMORY_CLASS_DYNAMICS );
		op.m_positionAndRotation[0] = position;
		op.m_positionAndRotation[1] = rotation.m_vec;
		getWorld()->queueOperation(op);
		return;
	}

	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_RW, this, HK_ACCESS_RW );
	setPositionAndRotation(position, rotation);
}

void hkpRigidBody::setLinearVelocityAsCriticalOperation(const hkVector4& newVel)
{
	hkCheckDeterminismUtil::checkMt(0xf0000230, newVel);

	if (getWorld() && getWorld()->areCriticalOperationsLocked())
	{
		hkWorldOperation::SetRigidBodyLinearVelocity op;
		op.m_rigidBody = this;
		newVel.store<3,HK_IO_NATIVE_ALIGNED>(&op.m_linearVelocity[0]);
		getWorld()->queueOperation(op);
		return;
	}

	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	setLinearVelocity(newVel);
}

void hkpRigidBody::setAngularVelocityAsCriticalOperation(const hkVector4& newVel)
{
	hkCheckDeterminismUtil::checkMt(0xf0000231, newVel);

	if (getWorld() && getWorld()->areCriticalOperationsLocked())
	{
		hkWorldOperation::SetRigidBodyAngularVelocity op;
		op.m_rigidBody = this;
		newVel.store<3,HK_IO_NATIVE_ALIGNED>(&op.m_angularVelocity[0]);
		getWorld()->queueOperation(op);
		return;
	}

	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	setAngularVelocity(newVel);
}

void hkpRigidBody::applyLinearImpulseAsCriticalOperation(const hkVector4& imp)
{
	hkCheckDeterminismUtil::checkMt(0xf0000210, imp);

	if (getWorld() && getWorld()->areCriticalOperationsLocked())
	{
		hkWorldOperation::ApplyRigidBodyLinearImpulse op;
		op.m_rigidBody = this;
		imp.store<3,HK_IO_NATIVE_ALIGNED>(&op.m_linearImpulse[0]);
		getWorld()->queueOperation(op);
		return;
	}

	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	applyLinearImpulse(imp);
}

void hkpRigidBody::applyPointImpulseAsCriticalOperation(const hkVector4& imp, const hkVector4& p)
{
	hkCheckDeterminismUtil::checkMt(0xf0000233, imp);
	hkCheckDeterminismUtil::checkMt(0xf0000234, p);

	if (getWorld() && getWorld()->areCriticalOperationsLocked())
	{
		hkWorldOperation::ApplyRigidBodyPointImpulse op;
		op.m_rigidBody = this;
		op.m_pointAndImpulse = hkAllocateChunk<hkVector4>( 2, HK_MEMORY_CLASS_DYNAMICS ); 
		op.m_pointAndImpulse[0] = p;
		op.m_pointAndImpulse[1] = imp;
		getWorld()->queueOperation(op);
		return;
	}

	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	applyPointImpulse(imp, p);
}

void hkpRigidBody::applyAngularImpulseAsCriticalOperation(const hkVector4& imp)
{
	hkCheckDeterminismUtil::checkMt(0xf0000234, imp);

	if (getWorld() && getWorld()->areCriticalOperationsLocked())
	{
		hkWorldOperation::ApplyRigidBodyAngularImpulse op;
		op.m_rigidBody = this;
		imp.store<3,HK_IO_NATIVE_ALIGNED>(&op.m_angularImpulse[0]);
		getWorld()->queueOperation(op);
		return;
	}

	HK_ACCESS_CHECK_WITH_PARENT( getWorld(), HK_ACCESS_IGNORE, this, HK_ACCESS_RW );
	applyAngularImpulse(imp);
}

hkBool hkpRigidBody::checkPerformance() const
{
	const hkpShape* shape = getCollidable()->getShape();

	if( !shape )
	{
		return false;
	}

	const int manyChildren = 100;
	hkpShapeType shapeType = shape->getType();
	hkBool isOk = true;

	if ( !isFixedOrKeyframed() )
	{
		// MOVING MOPP (with many children)
		if( shape->isBvTree()
			&& static_cast<const hkpBvTreeShape*>(shape)->getContainer()->getNumChildShapes() > manyChildren ) 
		{
			HK_WARN(0x2ff8c16d, "Rigid body at address " << this << " is a complex MOPP/bvtree shape (more than " << manyChildren << " children) \n" \
				"and has motion type other than fixed.\n" \
				"Note that this can cause a significant performance loss. \n" \
				"Consider simplifying the shape representation." );
			isOk = false;
		}
	}

	// THE CHECKS BELOW ARE DUPLICATED IN hkpCollidable::checkPerformance() TO BE SUPER SAFE. IF YOU CHANGE/ADD STUFF HERE
	// PLEASE SEE IF YOU NEED TO ALSO DO THE SAME IN hkpCollidable::checkPerformance() 

	// Mesh/triangle collection (without MOPP)
	if( shapeType == hkcdShapeType::TRIANGLE_COLLECTION && static_cast<const hkpShapeCollection*>(getCollidable()->getShape())->getNumChildShapes() > manyChildren )
	{
		HK_WARN(0x2ff8c16e, "Rigid body at address " << this << " has a mesh shape.\n" \
			"This can cause a significant performance loss.\n" \
			"The collection of triangle shapes (hkpMeshShape) should not be used for dynamic bodies.\n" \
			"You should consider building an hkpMoppBvTreeShape around the mesh.\n");
		isOk = false;
	}

	// Collection (without MOPP)
	if( (shapeType == hkcdShapeType::COLLECTION || shapeType == hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION)
		&& static_cast<const hkpShapeCollection*>(getCollidable()->getShape())->getNumChildShapes() > manyChildren )
	{
		HK_WARN(0x578cef50, "Rigid body at address " << this << " has a shape collection without a hkpBvTreeShape.\n" \
			"This can cause performance loss. To avoid getting this message\n" \
			"add a hkpBvTreeShape above this shape.");
		isOk = false;
	}

	// Transformed shape
	if( shapeType == hkcdShapeType::TRANSFORM)
	{
		HK_WARN(0x2ff8c16f, "Rigid body at address " << this << " has a transform shape as the root shape.\n" \
			"This can cause a significant performance loss. To avoid getting this message\n" \
			"compose the transform into the collidable and remove the transform shape.\n" \
			"Also, not that the transform shape won't work in the new split collision pipeline.\n" \
			"Please see the 'hkpTransformShape' documentation in the User Guide for more information.\n");
		isOk = false;
	}

	// List (with many children)
	if( shapeType == hkcdShapeType::LIST 
		&& static_cast<const hkpListShape*>(shape)->getNumChildShapes() > manyChildren ) 
	{
		HK_WARN(0x2ff8c16c, "Rigid body at address " << this << " has an hkpListShape with > " << manyChildren << " children.\n" \
			"This can cause a significant performance loss when the shape is collided e.g. with another complex hkpListShape.\n" \
			"When using hkpListShape with many children consider building an hkpMoppBvTreeShape around it.\n");
		isOk = false;
	}
	
	return isOk;
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
