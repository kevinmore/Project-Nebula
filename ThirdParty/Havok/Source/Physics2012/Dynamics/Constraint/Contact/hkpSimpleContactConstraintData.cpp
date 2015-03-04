/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics/ConstraintSolver/Solve/hkpSolve.h>
#include <Physics2012/Internal/Solver/SimpleConstraints/hkpSimpleConstraintUtil.h>
#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryIn.h>

#include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgr.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>

#include <Physics2012/Dynamics/Constraint/Response/hkpSimpleCollisionResponse.h>
#include <Physics2012/Dynamics/Constraint/Contact/hkpSimpleContactConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Contact/hkpContactImpulseLimitBreachedListener.h>
#include <Physics2012/Dynamics/Constraint/hkpConstraintOwner.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldConstraintUtil.h>
#include <Physics2012/Dynamics/Entity/Util/hkpEntityCallbackUtil.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>

#include <Physics2012/Dynamics/Constraint/Contact/hkpSimpleContactConstraintUtil.h>

#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>


#if defined(HK_PLATFORM_HAS_SPU)	// see HVK-3955
hkpSimpleConstraintContactMgr* hkpContactImpulseLimitBreachedListenerInfo::getContactMgr() const
{
	const hkLong offset = hkLong(-HK_OFFSET_OF( hkpSimpleConstraintContactMgr, m_constraint ));
	return (hkpSimpleConstraintContactMgr*)hkAddByteOffset( m_data.m_single.m_constraintInstance, offset );
}
#endif

hkContactPointId hkpSimpleContactConstraintData::allocateContactPoint( hkpConstraintOwner& constraintOwner, hkContactPoint** cpOut, hkpContactPointProperties** cpPropsOut)
{
	hkpSimpleContactConstraintAtom* localAtom = HK_GET_LOCAL_CONTACT_ATOM(m_atom);

	localAtom->m_info.m_flags |= hkpSimpleContactConstraintDataInfo::HK_FLAG_AREA_CHANGED;

	const int newContactIndex = localAtom->m_numContactPoints;

	const hkpSimpleContactConstraintAtom* originalAtomOnPpu = m_atom;

	hkPadSpu<bool> atomReallocated = false;
	m_atom = hkpSimpleContactConstraintAtomUtil::expandOne(m_atom, atomReallocated);
	localAtom = HK_GET_LOCAL_CONTACT_ATOM(m_atom);

	hkContactPoint* dcp           = &localAtom->getContactPoints()[newContactIndex];
	int cpiStriding = localAtom->getContactPointPropertiesStriding();
	hkpContactPointPropertiesStream* cpi = hkAddByteOffset( localAtom->getContactPointPropertiesStream(), newContactIndex * cpiStriding );
	hkpContactPointPropertiesStream* cpi_less1 = hkAddByteOffset(cpi, -cpiStriding);

	cpi->asProperties()->init();

	hkUchar& flags = const_cast<hkUchar&>(cpi->asProperties()->m_flags);
	flags = hkpContactPointProperties::CONTACT_IS_NEW;
	if ( newContactIndex > 0 &&  !(cpi_less1->asProperties()->m_flags & hkpContactPointProperties::CONTACT_USES_SOLVER_PATH2) && (!cpi_less1->asProperties()->m_maxImpulse.m_value) )
	{
		flags |= hkpContactPointProperties::CONTACT_USES_SOLVER_PATH2;
	}

	*cpOut = dcp;
	*cpPropsOut = cpi->asProperties();

	hkpConstraintInfo info;
	{
		info.clear();
		int newSize = newContactIndex + 1;
		info.m_maxSizeOfSchema = hkpJacobianSchemaInfo::Header::Sizeof 
							+ (newSize >> 1) * hkpJacobianSchemaInfo::PairContact::Sizeof 
							+ (newSize & 1) * hkpJacobianSchemaInfo::SingleContact::Sizeof 
							+ hkpJacobianSchemaInfo::Friction3D::Sizeof;

		if ( newContactIndex == 1 )		// angular friction added 
		{
			info.add( hkpJacobianSchemaInfo::Friction3D::Sizeof - hkpJacobianSchemaInfo::Friction2D::Sizeof, 1, (1+0) ); // tempElem for friction doesn't change
		}

		if ( newSize & 1 ) // size after allocation of the new point
		{
			info.add( hkpJacobianSchemaInfo::SingleContact::Sizeof, 1, 1 );
		}
		else
		{
			info.add( hkpJacobianSchemaInfo::PairContact::Sizeof - hkpJacobianSchemaInfo::SingleContact::Sizeof, 1, 1 );
		}
	}

	constraintOwner.addConstraintInfo( m_constraint, info );

	if (atomReallocated.val())
	{
		hkpWorldConstraintUtil::updateFatherOfMovedAtom( m_constraint, originalAtomOnPpu, m_atom, localAtom->m_sizeOfAllAtoms );
		m_atomSize = localAtom->m_sizeOfAllAtoms;
	}

	return hkContactPointId(m_idMgrA.newId( newContactIndex ));
}



HK_COMPILE_TIME_ASSERT( int(hkpJacobianSchemaInfo::Friction3D::Sizeof) >= int(hkpJacobianSchemaInfo::Friction2D::Sizeof));

#if !defined(HK_PLATFORM_SPU)
hkpSimpleContactConstraintData::hkpSimpleContactConstraintData(hkpConstraintInstance* constraint, hkpRigidBody* bodyA, hkpRigidBody* bodyB)
{
	m_constraint = constraint;

	const int extraDataA = bodyA->m_numShapeKeysInContactPointProperties;
	const int extraDataB = bodyB->m_numShapeKeysInContactPointProperties;
	const int maxSizeAllContactPointsAndProperties = HK_MAX_CONTACT_POINT * (sizeof(hkContactPoint) + sizeof(hkpContactPointProperties));
	const int sizeOfOneContactPointAndExtededProperties = sizeof(hkContactPoint) + HK_NEXT_MULTIPLE_OF(sizeof(hkReal),sizeof(hkpContactPointProperties) + sizeof(hkpContactPointProperties::UserData) * (extraDataA + extraDataB));
	const int maxNumContactPoints = (hkUint16)(maxSizeAllContactPointsAndProperties / sizeOfOneContactPointAndExtededProperties);

#if !defined(HK_PLATFORM_HAS_SPU)
	m_atom = hkpSimpleContactConstraintAtomUtil::allocateAtom(1, extraDataA, extraDataB, maxNumContactPoints);
#else
	m_atom = hkpSimpleContactConstraintAtomUtil::allocateAtom(4, extraDataA, extraDataB, maxNumContactPoints);
#endif
	
	m_atom->m_info.init();
	m_atomSize = m_atom->m_sizeOfAllAtoms;
}
#endif

#if !defined(HK_PLATFORM_SPU)
void hkpSimpleContactConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const 
{
	infoOut.m_atoms = m_atom;
	infoOut.m_sizeOfAllAtoms = m_atom->m_sizeOfAllAtoms;
	infoOut.clear();

	m_atom->addToConstraintInfo(infoOut);
}

void hkpSimpleContactConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	// all data handled internally
	infoOut.m_sizeOfExternalRuntime = 0;
	infoOut.m_numSolverResults = 0;
}

hkpSolverResults* hkpSimpleContactConstraintData::getSolverResults( hkpConstraintRuntime* runtime )
{
	return HK_NULL;
}


hkpSimpleConstraintContactMgr* hkpSimpleContactConstraintData::getSimpleConstraintContactMgr() const
{
	// The pointer manipulation below should be safe since the constructor of this object's class is 
	// restricted to the appropriate manager.
	const int offset = HK_OFFSET_OF( hkpSimpleConstraintContactMgr, m_contactConstraintData );
	const void *const cmgr = hkAddByteOffsetConst( this, -offset );
	void *const cmgrNC = const_cast<void*>( cmgr );
	hkpSimpleConstraintContactMgr *const contactMgr = reinterpret_cast<hkpSimpleConstraintContactMgr*>( cmgrNC );
	return contactMgr;
}

#ifdef HK_DEBUG
// Confirm that the user re-synced the accumulator and body motion velocities if they changed them during a callback.
static void checkVelocitiesUpdated( hkpRigidBody* body, hkpVelocityAccumulator* accum, hkVector4& linOld, hkVector4& angOld )
{
	hkpMotion* motion = body->getMotion();
	// Only check if the user modified one of the body's velocities.
	const hkSimdReal eps = hkSimdReal::fromFloat(1e-3f);
	if ( ( !linOld.allEqual<3>( motion->m_linearVelocity, eps ) ) || ( !angOld.allEqual<3>( motion->m_angularVelocity, eps ) ) )
	{
		HK_ASSERT2( 0x45f29ace, motion->m_linearVelocity.allEqual<3>( accum->m_linearVel, eps ), "You modified the linear velocity of a body during a callback without calling accessVelocities/updateVelocities." );
		if ( ( motion->getType() == hkpMotion::MOTION_BOX_INERTIA ) || ( motion->getType() == hkpMotion::MOTION_THIN_BOX_INERTIA ) )
		{
			hkVector4 angVel;
			angVel._setRotatedDir( motion->getTransform().getRotation(), accum->m_angularVel );
			HK_ASSERT2( 0x45f29ace, motion->m_angularVelocity.allEqual<3>( angVel, eps ), "You modified the angular velocity of a body during a callback without calling accessVelocities/updateVelocities." );
		}
		else
		{
			HK_ASSERT2( 0x45f29ace, motion->m_angularVelocity.allEqual<3>( accum->m_angularVel, eps ), "You modified the angular velocity of a body during a callback without calling accessVelocities/updateVelocities." );
		}
	}
}
#endif

void hkSimpleContactConstraintData_fireCallbacks( hkpSimpleContactConstraintData* constraintData, const hkpConstraintQueryIn* in, hkpSimpleContactConstraintAtom* atom, hkpContactPointEvent::Type type )
{
	HK_ASSERT2( 0x10de67ab, ( type == hkpContactPointEvent::TYPE_MANIFOLD ) || ( type == hkpContactPointEvent::TYPE_EXPAND_MANIFOLD ), "This method is only intended for manifold events" );
	hkpConstraintInstance *const constraintInstance = in->m_constraintInstance;
	hkpSimpleConstraintContactMgr *const mgr = constraintData->getSimpleConstraintContactMgr();

	hkpRigidBody *const bodyA = constraintInstance->getRigidBodyA();
	hkpRigidBody *const bodyB = constraintInstance->getRigidBodyB();
	hkpWorld* world = bodyA->getWorld();
	hkpVelocityAccumulator *const accA = const_cast< hkpVelocityAccumulator* >( in->m_bodyA.val() );
	hkpVelocityAccumulator *const accB = const_cast< hkpVelocityAccumulator* >( in->m_bodyB.val() );

	const hkBool callbacksForFullManifold = 0 < ( constraintInstance->m_internal->m_callbackRequest & hkpConstraintAtom::CALLBACK_REQUEST_CONTACT_POINT_CALLBACK );

	HK_ON_DEBUG_MULTI_THREADING( if ( !bodyA->isFixed() ) { HK_ACCESS_CHECK_WITH_PARENT( bodyA->getWorld(), HK_ACCESS_IGNORE, bodyA, HK_ACCESS_RW); } ) ;
	HK_ON_DEBUG_MULTI_THREADING( if ( !bodyB->isFixed() ) { HK_ACCESS_CHECK_WITH_PARENT( bodyB->getWorld(), HK_ACCESS_IGNORE, bodyB, HK_ACCESS_RW); } ) ;

	hkpContactPointPropertiesStream* cpp = atom->getContactPointPropertiesStream();
	hkpContactPointPropertiesStream* last_cpp = HK_NULL;
	const int cppStriding = atom->getContactPointPropertiesStriding();
	hkContactPoint* cp = atom->getContactPoints();
	const int nA = atom->m_numContactPoints;
// 	HK_REPORT("nA "<<nA);

	// Iterate over the atom's contact points
	for (int cindex = nA-1; cindex >=0 ; cp++, last_cpp = cpp, cpp = hkAddByteOffset(cpp, cppStriding), cindex-- )
	{
		hkUint8& flags = cpp->asProperties()->m_flags;

		if ( flags & hkContactPointMaterial::CONTACT_IS_NEW )
		{
			hkSimdReal projectedVel = hkpSimpleContactConstraintUtil::calculateSeparatingVelocity( bodyA, bodyB, in->m_bodyA->getCenterOfMassInWorld(), in->m_bodyB->getCenterOfMassInWorld(), cp );
// 			HK_REPORT("projVel "<<projectedVel);
// 			const hkVector4& cpPos = cp->getPosition();
// 			HK_REPORT("cpPos "<<cpPos(0)<<" "<<cpPos(1)<<" "<<cpPos(2)<<" "<<cpPos(3));
			// Fire contact point callbacks
			{
				// Attempt to catch a modification of a bodies velocities which isn't followed by a call to updateVelocities().
				HK_ON_DEBUG( hkVector4 linAold; linAold = bodyA->getMotion()->m_linearVelocity );
				HK_ON_DEBUG( hkVector4 linBold; linBold = bodyB->getMotion()->m_linearVelocity );
				HK_ON_DEBUG( hkVector4 angAold; angAold = bodyA->getMotion()->m_angularVelocity );
				HK_ON_DEBUG( hkVector4 angBold; angBold = bodyB->getMotion()->m_angularVelocity );

				hkpShapeKey *const shapeKeys = reinterpret_cast< hkpShapeKey* >( cpp->asProperties()->getStartOfExtendedUserData( atom ) );
				hkReal projectedVelScalar; projectedVel.store<1>(&projectedVelScalar);
				hkpContactPointEvent event( hkpCollisionEvent::SOURCE_WORLD, bodyA, bodyB, mgr,
					type, 
					cp, cpp->asProperties(), 
					&projectedVelScalar, HK_NULL, 
					callbacksForFullManifold, ( cindex == nA - 1 ), ( cindex == 0 ),
					shapeKeys,
					accA, accB );

				hkpWorldCallbackUtil::fireContactPointCallback( world, event );

				event.m_source = hkpCollisionEvent::SOURCE_A;
				hkpEntityCallbackUtil::fireContactPointCallback( bodyA, event );

				event.m_source = hkpCollisionEvent::SOURCE_B;
				hkpEntityCallbackUtil::fireContactPointCallback( bodyB, event );

				projectedVel.load<1>(&projectedVelScalar);
				HK_ON_DEBUG( checkVelocitiesUpdated( bodyA, accA, linAold, angAold ) );
				HK_ON_DEBUG( checkVelocitiesUpdated( bodyB, accB, linBold, angBold ) );
			}

			if ( ( cindex < nA - 1 ) && ( last_cpp->asProperties()->m_maxImpulse.m_value ) )
			{
				flags &= ~hkContactPointMaterial::CONTACT_USES_SOLVER_PATH2;
			}

			// Skip initial collision response if both bodies are fixed/keyframed or the contact is disabled
			hkBool32 hasDynamicComponent = (	accA->m_invMasses.notEqualZero().anyIsSet() |
											accB->m_invMasses.notEqualZero().anyIsSet() );
			if ( !hasDynamicComponent || ( flags & hkContactPointMaterial::CONTACT_IS_DISABLED ) || mgr->isConstraintDisabled() )
			{
				cpp->asProperties()->m_impulseApplied = hkReal(0);
				cpp->asProperties()->m_internalSolverData = hkReal(0);
				cpp->asProperties()->m_internalDataA = hkReal(0);
			}
			else
			{
				// Do initial collision response
				const hkpWorldDynamicsStepInfo& env = *reinterpret_cast<const hkpWorldDynamicsStepInfo*>(world->getCollisionInput()->m_dynamicsInfo);
				const hkSimdReal restitution = hkSimdReal::fromFloat(cpp->asProperties()->getRestitution());
				const hkSimdReal MIN_RESTITUTION_TO_USE_CORRECT_RESPONSE = hkSimdReal::fromFloat(hkReal(0.3f));
				hkSimdReal restingVel; restingVel.load<1>(&env.m_solverInfo.m_contactRestingVelocity);
				if ( projectedVel < -restingVel && restitution > MIN_RESTITUTION_TO_USE_CORRECT_RESPONSE )
				{
					// Accurate version. Solve and update the body velocities now
					hkpSimpleConstraintUtilCollideParams params;
					{
						params.m_friction  = cpp->asProperties()->getFriction();
						restitution.store<1>(&params.m_restitution);
						params.m_extraSeparatingVelocity = hkReal(0);
						params.m_extraSlope = hkReal(0);
						projectedVel.store<1>(&params.m_externalSeperatingVelocity);
						params.m_maxImpulse = HK_REAL_MAX;
						if ( !cpp->asProperties()->m_maxImpulse.isZero() )
						{
							params.m_maxImpulse = cpp->asProperties()->getMaxImpulsePerStep();
						}
					}

					hkpSimpleCollisionResponse::SolveSingleOutput2 out2;
					hkpVelocityAccumulator* accuA = const_cast<hkpVelocityAccumulator*>(in->m_bodyA.val());
					hkpVelocityAccumulator* accuB = const_cast<hkpVelocityAccumulator*>(in->m_bodyB.val());
					hkpSimpleCollisionResponse::solveSingleContact2( constraintData, *cp, params, bodyA, bodyB, accuA, accuB, out2 );

					if ( params.m_contactImpulseLimitBreached )
					{
						hkpContactImpulseLimitBreachedListenerInfo breachInfo;
						breachInfo.set( in->m_constraintInstance, cpp->asProperties(), cp, false );
						hkpWorldCallbackUtil::fireContactImpulseLimitBreached( world, &breachInfo, 1 );
					}

					
					
					cpp->asProperties()->m_impulseApplied = hkReal(0); // out2.m_impulse * 0.5f;
					cpp->asProperties()->m_internalDataA = hkReal(0); 
				}
				else
				{
					// Quick version. Don't update velocities, just prepare the solver.

					
					hkSimdReal sumInvMass = bodyA->getRigidMotion()->getMassInv() + bodyB->getRigidMotion()->getMassInv();
					hkSimdReal mass; mass.setReciprocal( sumInvMass + hkSimdReal::fromFloat(hkReal(1e-10f)) );
					(mass * hkSimdReal::fromFloat(hkReal(-0.2f)) * projectedVel * ( hkSimdReal_1 + restitution )).store<1>(&cpp->asProperties()->m_impulseApplied);

					
					hkSimdReal errorTerm; errorTerm.load<1>((const hkReal*)&(in->m_subStepDeltaTime.ref()));
					errorTerm.mul(restitution * projectedVel);
					errorTerm.mul(hkSimdReal::fromFloat(hkReal(-1.3f)));
					errorTerm.store<1>(&cpp->asProperties()->m_internalSolverData);

					
					hkSimdReal dist = cp->getDistanceSimdReal() - errorTerm;
					dist.zeroIfFalse(restitution.greaterZero());
					dist.store<1>(&cpp->asProperties()->m_internalDataA);
				}
			}
			flags &= ~hkpContactPointProperties::CONTACT_IS_NEW;
		}
		else if ( callbacksForFullManifold )
		{
			// Fire contact point callbacks for non-new contacts
			{
				// Attempt to catch a modification of a bodies velocities which isn't followed by a call to updateVelocities().
				HK_ON_DEBUG( hkVector4 linAold; linAold = bodyA->getMotion()->m_linearVelocity );
				HK_ON_DEBUG( hkVector4 linBold; linBold = bodyB->getMotion()->m_linearVelocity );
				HK_ON_DEBUG( hkVector4 angAold; angAold = bodyA->getMotion()->m_angularVelocity );
				HK_ON_DEBUG( hkVector4 angBold; angBold = bodyB->getMotion()->m_angularVelocity );

				hkpShapeKey *const shapeKeys = reinterpret_cast< hkpShapeKey* >( cpp->asProperties()->getStartOfExtendedUserData( atom ) );
				hkpContactPointEvent event( hkpCollisionEvent::SOURCE_WORLD, bodyA, bodyB, constraintData->getSimpleConstraintContactMgr(),
					type, 
					cp, cpp->asProperties(), 
					HK_NULL, HK_NULL, 
					true, cindex == nA - 1, cindex == 0,
					shapeKeys,
					accA, accB );

				hkpWorldCallbackUtil::fireContactPointCallback( world, event );

				event.m_source = hkpCollisionEvent::SOURCE_A;
				hkpEntityCallbackUtil::fireContactPointCallback( bodyA, event );

				event.m_source = hkpCollisionEvent::SOURCE_B;
				hkpEntityCallbackUtil::fireContactPointCallback( bodyB, event );

				HK_ON_DEBUG( checkVelocitiesUpdated( bodyA, accA, linAold, angAold ) );
				HK_ON_DEBUG( checkVelocitiesUpdated( bodyB, accB, linBold, angBold ) );
			}

			if ( ( cindex < nA - 1 ) && ( last_cpp->asProperties()->m_maxImpulse.m_value ) )
			{
				flags &= ~hkContactPointMaterial::CONTACT_USES_SOLVER_PATH2;
			}
		}
	}

	// clear the callback request flag once the callbacks have been fired
	constraintInstance->m_internal->m_callbackRequest &= ~( hkpConstraintAtom::CALLBACK_REQUEST_CONTACT_POINT_CALLBACK | hkpConstraintAtom::CALLBACK_REQUEST_NEW_CONTACT_POINT );
}


void hkpSimpleContactConstraintData::collisionResponseBeginCallback( const hkContactPoint& cp, struct hkpSimpleConstraintInfoInitInput& inA, struct hkpBodyVelocity& velA, hkpSimpleConstraintInfoInitInput& inB, hkpBodyVelocity& velB)
{
	hkpConstraintInstance* instance = m_constraint;
	if ( !instance->m_constraintModifiers )
	{
		return;
	}

	// iterate over all atom modifiers
	hkpConstraintAtom* atom = instance->m_constraintModifiers;

	while ( atom->isModifierType() )
	{
		switch ( atom->getType())
		{
		case hkpConstraintAtom::TYPE_MODIFIER_MASS_CHANGER:
			{
				hkpMassChangerModifierConstraintAtom* c = reinterpret_cast<hkpMassChangerModifierConstraintAtom*>(atom);
				c->collisionResponseBeginCallback( cp, inA, velA, inB, velB );
				break;
			}

		case hkpConstraintAtom::TYPE_MODIFIER_SOFT_CONTACT:
			{
				hkpSoftContactModifierConstraintAtom* c = reinterpret_cast<hkpSoftContactModifierConstraintAtom*>(atom);
				c->collisionResponseBeginCallback( cp, inA, velA, inB, velB );
				break;
			}

		case hkpConstraintAtom::TYPE_MODIFIER_MOVING_SURFACE:
			{
				hkpMovingSurfaceModifierConstraintAtom* c = reinterpret_cast<hkpMovingSurfaceModifierConstraintAtom*>(atom);
				c->collisionResponseBeginCallback( cp, inA, velA, inB, velB );
				break;
			}
		case hkpConstraintAtom::TYPE_MODIFIER_CENTER_OF_MASS_CHANGER:
			{
				hkpCenterOfMassChangerModifierConstraintAtom* c = reinterpret_cast<hkpCenterOfMassChangerModifierConstraintAtom*>(atom);
				c->collisionResponseBeginCallback( cp, inA, velA, inB, velB );
				break;
			}
		default:
			{
				break;
			}
		}
		atom = static_cast<hkpModifierConstraintAtom*>(atom)->m_child;
	}
}

void hkpSimpleContactConstraintData::collisionResponseEndCallback( const hkContactPoint& cp, hkReal impulseApplied, struct hkpSimpleConstraintInfoInitInput& inA, struct hkpBodyVelocity& velA, hkpSimpleConstraintInfoInitInput& inB, hkpBodyVelocity& velB)
{
	hkpConstraintInstance* instance = m_constraint;

	if ( !instance->m_constraintModifiers )
	{
		return;
	}


	// iterate over all atom modifiers and call them in the reverse order
	hkInplaceArray<hkpConstraintAtom*,16> atoms;
	{
		for ( hkpConstraintAtom* atom = instance->m_constraintModifiers; atom->isModifierType(); atom = static_cast<hkpModifierConstraintAtom*>(atom)->m_child )
		{
			atoms.pushBack( atom );
		}
	}

	for (int i= atoms.getSize()-1; i>=0; i--)
	{
		hkpConstraintAtom* atom = atoms[i];
		switch ( atom->getType())
		{
		case hkpConstraintAtom::TYPE_MODIFIER_MASS_CHANGER:
			{
				hkpMassChangerModifierConstraintAtom* c = static_cast<hkpMassChangerModifierConstraintAtom*>(atom);
				c->collisionResponseEndCallback( cp, impulseApplied, inA, velA, inB, velB );
				break;
			}

		case hkpConstraintAtom::TYPE_MODIFIER_SOFT_CONTACT:
			{
				hkpSoftContactModifierConstraintAtom* c = static_cast<hkpSoftContactModifierConstraintAtom*>(atom);
				c->collisionResponseEndCallback( cp, impulseApplied, inA, velA, inB, velB );
				break;
			}

		case hkpConstraintAtom::TYPE_MODIFIER_MOVING_SURFACE:
			{
				hkpMovingSurfaceModifierConstraintAtom* c = static_cast<hkpMovingSurfaceModifierConstraintAtom*>(atom);
				c->collisionResponseEndCallback( cp, impulseApplied, inA, velA, inB, velB );
				break;
			}
		case hkpConstraintAtom::TYPE_MODIFIER_CENTER_OF_MASS_CHANGER:
			{
				hkpCenterOfMassChangerModifierConstraintAtom* c = reinterpret_cast<hkpCenterOfMassChangerModifierConstraintAtom*>(atom);
				c->collisionResponseEndCallback( cp, impulseApplied, inA, velA, inB, velB );
				break;
			}

		default:
			{
				break;
			}
		}
	}
}



hkBool hkpSimpleContactConstraintData::isValid() const
{
	return true;
}

int hkpSimpleContactConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_CONTACT;
}
#endif // !defined(HK_PLATFORM_SPU)


HK_COMPILE_TIME_ASSERT( sizeof( hkpSimpleContactConstraintData ) < 256 );

HK_COMPILE_TIME_ASSERT( hkpJacobianSchemaInfo::PairContact::Sizeof >= 2* hkpJacobianSchemaInfo::SingleContact::Sizeof );

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
