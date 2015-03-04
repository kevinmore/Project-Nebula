/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>

#include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgr.h>
#include <Physics2012/Collide/Agent/hkpProcessCdPoint.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>
#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Entity/Util/hkpEntityCallbackUtil.h>

#include <Physics2012/Dynamics/Constraint/Contact/hkpSimpleContactConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Response/hkpSimpleCollisionResponse.h>

#include <Physics2012/Dynamics/Collide/hkpSimpleConstraintContactMgr.h>
#include <Physics2012/Dynamics/Collide/Deprecated/Dispatch/hkpCollideCallbackDispatcher.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldCallbackUtil.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldConstraintUtil.h>
#include <Physics2012/Dynamics/World/Simulation/Continuous/hkpContinuousSimulation.h>

#include <Physics2012/Internal/Solver/SimpleConstraints/hkpSimpleConstraintUtil.h>

// Needed for the class reflection
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkTypeInfo.h>



#if defined(HK_PLATFORM_SPU)
#	include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgrSpu.inl> // include this after we include the actual contact manager!
#endif

#ifndef HK_PLATFORM_SPU
HK_REFLECTION_DEFINE_STUB_VIRTUAL(hkpSimpleConstraintContactMgr,hkReferencedObject);
#endif

#if !defined(HK_PLATFORM_SPU)
hkpConstraintInstance* hkpSimpleConstraintContactMgr::getConstraintInstance()
{
	return &m_constraint;
}

const hkpConstraintInstance* hkpSimpleConstraintContactMgr::getConstraintInstance() const
{
	return &m_constraint;
}
#endif

#if !defined(HK_PLATFORM_SPU)
hkpSimpleConstraintContactMgr::hkpSimpleConstraintContactMgr( hkpWorld *sm, hkpRigidBody *bodyA, hkpRigidBody *bodyB ): hkpDynamicsContactMgr( hkpContactMgr::TYPE_SIMPLE_CONSTRAINT_CONTACT_MGR ),
	m_contactConstraintData( &m_constraint, bodyA, bodyB ), //m_constraint(bodyA, bodyB, &m_contactConstraintData, hkpConstraintInstance::PRIORITY_PSI, hkpConstraintInstance::DO_NOT_ADD_REFERENCES__FOR_INTERNAL_USE_ONLY) // we don't want to add references to rigidBodies and constraintDatas
												m_constraint(hkpConstraintInstance::PRIORITY_PSI)
{
	// Init constraint instance; don't add references to data and entities -- similarly those references have to be set to
	// HK_NULL before the constraintInstance destructor is called (to avoid removeReference() being called automatically.)
	m_constraint.m_data = &m_contactConstraintData;
	m_constraint.m_entities[0] = bodyA;
	m_constraint.m_entities[1] = bodyB;
	HK_ASSERT2(0xf0fe4356, bodyA, "EntityA not set.");
	m_constraint.m_uid = 0xffffffff;

	//

	m_world = sm;
	// This causes callbacks for the full contact manifold to be fired for the first frame, regardless of the bodies'
	// m_contactPointCallbackDelay. But this doesn't matter, because the manifold for the first frame will contain
	// only new contact points.
	m_contactPointCallbackDelay = 0;
	m_reservedContactPoints = 0;

	//  memsize 0 is a special case for objects that can't have their dtor called.
	m_contactConstraintData.m_memSizeAndFlags = 0;
	m_constraint.m_memSizeAndFlags = 0;

	hkpCollidableQualityType a = bodyA->getCollidable()->getQualityType();
	hkpCollidableQualityType b = bodyB->getCollidable()->getQualityType();

	hkpCollisionDispatcher* dispatcher = sm->getCollisionDispatcher();
	hkpCollisionDispatcher::CollisionQualityIndex index = dispatcher->getCollisionQualityIndex( a, b );
	hkpCollisionQualityInfo* qualityInfo = dispatcher->getCollisionQualityInfo( index );
	int constraintPriority = qualityInfo->m_constraintPriority;

	m_constraint.setPriority( hkpConstraintInstance::ConstraintPriority(constraintPriority) );

	/// BETA: Rolling friction is a work in progress. It is experimental code and has significant behavior artifacts.
	const hkReal rollingMultiplierA = bodyA->getMaterial().getRollingFrictionMultiplier();
	const hkReal rollingMultiplierB = bodyB->getMaterial().getRollingFrictionMultiplier();
	const hkReal combinedMultiplier = hkpMaterial::getCombinedRollingFrictionMultiplier(rollingMultiplierA, rollingMultiplierB);
	setRollingFrictionMultiplier(combinedMultiplier);
}
#endif

#if !defined(HK_PLATFORM_SPU)
hkpSimpleConstraintContactMgr::~hkpSimpleConstraintContactMgr()
{
	{
		int size = m_contactConstraintData.m_atom->m_numContactPoints;
		if ( size )
		{
			/*
			HK_ASSERT2(0x739cc779,  0, "A contact mgr is deleted before all the contact points are removed.\n" \
				"Did you try to change the collision filter from a callback? You need to\n" \
				"do that outside the callback.");
			*/
			hkpWorldOperationUtil::removeConstraintImmediately( m_world, &m_constraint );
		}

		// Simply set the entities and constraintData pointers to zero so that the ~hkpConstraintInstance does not remove references.
		// (references are not thread safe )
		m_constraint.m_entities[0] = HK_NULL;
		m_constraint.m_entities[1] = HK_NULL;
		m_constraint.m_data = HK_NULL;
	}
}
#endif

hkResult hkpSimpleConstraintContactMgr::reserveContactPointsImpl( int numPoints )
{
	hkpSimpleContactConstraintAtom* localAtom = HK_GET_LOCAL_CONTACT_ATOM(m_contactConstraintData.m_atom);

	int size = localAtom->m_numContactPoints;
	if ( size + numPoints + m_reservedContactPoints > localAtom->m_maxNumContactPoints - 2 )
	{
		HK_WARN_ONCE( 0xf0e31234, "Maximum number of ContactPoints reached, dropping points Note: This message is fired only once and it means that one object has too many triangles in a specific area" );
		return HK_FAILURE;
	}
	m_reservedContactPoints = hkUchar(m_reservedContactPoints + numPoints);
	return HK_SUCCESS;
}

void hkpSimpleConstraintContactMgr::removeToiImpl( class hkCollisionConstraintOwner& constraintOwner, hkpContactPointProperties& properties )
{
#if ! defined (HK_PLATFORM_SPU)
	hkpRigidBody* bodyA = m_constraint.getRigidBodyA();
	hkpRigidBody* bodyB = m_constraint.getRigidBodyB();
#else
	extern hkpEntity* g_spuCollisionEventsEntityAOnSpu;
	extern hkpEntity* g_spuCollisionEventsEntityBOnSpu;
	hkpRigidBody* bodyA = (hkpRigidBody*)g_spuCollisionEventsEntityAOnSpu;
	hkpRigidBody* bodyB = (hkpRigidBody*)g_spuCollisionEventsEntityBOnSpu;
#endif

	// Fire TOI-point removed
	hkpContactPointRemovedEvent removedEvent( HK_INVALID_CONTACT_POINT, this, &constraintOwner, &properties, bodyA, bodyB );
	hkFireContactPointRemovedCallback ( m_world, bodyA, bodyB, removedEvent );
}

#if !defined(HK_PLATFORM_SPU)

hkBool hkpSimpleConstraintContactMgr::fireCallbacksForEarliestToi( struct hkpToiEvent& event, hkReal& rotateNormal )
{
	hkpShapeKey* shapeKeys = reinterpret_cast<hkpShapeKey*>( &event.m_extendedUserDatas );
	hkpContactPointEvent cpEvent( hkpCollisionEvent::SOURCE_WORLD, static_cast<hkpRigidBody*>( event.m_entities[0] ), static_cast<hkpRigidBody*>( event.m_entities[1] ), this,
		hkpContactPointEvent::TYPE_TOI,
		&event.m_contactPoint, &event.m_properties, 
		&event.m_seperatingVelocity, &rotateNormal, 
		false, false, false,
		shapeKeys,
		HK_NULL, HK_NULL );
	hkpWorld* world = event.m_entities[0]->getWorld();
	hkpWorldCallbackUtil::fireContactPointCallback( world, cpEvent );

	cpEvent.m_source = hkpCollisionEvent::SOURCE_A;
	hkpEntityCallbackUtil::fireContactPointCallback( event.m_entities[0], cpEvent );

	cpEvent.m_source = hkpCollisionEvent::SOURCE_B;
	hkpEntityCallbackUtil::fireContactPointCallback( event.m_entities[1], cpEvent );

	return !( isConstraintDisabled() || ( event.m_properties.m_flags & hkContactPointMaterial::CONTACT_IS_DISABLED ) );
}


void hkpSimpleConstraintContactMgr::confirmToi( struct hkpToiEvent& event, hkReal rotateNormal, class hkArray<class hkpEntity*>& outToBeActivated )
{
	hkpWorld* world = event.m_entities[0]->getWorld();

	if ( ! ( event.m_properties.m_flags & hkContactPointMaterial::CONTACT_IS_DISABLED ) )
	{
		// Do simple response and mark bodies active/not active;
		hkpRigidBody* body0 = static_cast<hkpRigidBody*>(event.m_entities[0]);
		hkpRigidBody* body1 = static_cast<hkpRigidBody*>(event.m_entities[1]);

		if ( !body0->isFixedOrKeyframed() || !body1->isFixedOrKeyframed() )
		{
			// Process the TOI-event contact using hkpSimpleCollisionResponse
			// ( doSimpleCollisionResponse does not require the bodies to backstepped now )
			hkLs_doSimpleCollisionResponse( world, event, rotateNormal, outToBeActivated );
		}
	}
}
#endif




hkpRigidBody* hkpSimpleConstraintContactMgr::setContactPointProperties( const hkpCdBody& a, const hkpCdBody& b, int maxNumExtraDataInEvent, hkpContactPointProperties* cpi )
{
#if ! defined (HK_PLATFORM_SPU)
	hkpRigidBody* sortedEa = m_constraint.getRigidBodyA();
	hkpRigidBody* sortedEb = m_constraint.getRigidBodyB();
#else
	extern hkpEntity* g_spuCollisionEventsEntityAOnSpu;
	extern hkpEntity* g_spuCollisionEventsEntityBOnSpu;
	hkpRigidBody* sortedEa = (hkpRigidBody*)g_spuCollisionEventsEntityAOnSpu;
	hkpRigidBody* sortedEb = (hkpRigidBody*)g_spuCollisionEventsEntityBOnSpu;
#endif

	const hkpMaterial& materialA = sortedEa->getMaterial();
	const hkpMaterial& materialB = sortedEb->getMaterial();

	cpi->setFriction(    hkpMaterial::getCombinedFriction(    materialA.getFriction(), materialB.getFriction() ) );
	cpi->setRestitution( hkpMaterial::getCombinedRestitution( materialA.getRestitution(), materialB.getRestitution() ) );
	cpi->m_maxImpulse.m_value = 0;

	hkpRigidBody* ea = static_cast<hkpRigidBody*>(a.getRootCollidable()->getOwner());
	{
		hkpSimpleContactConstraintAtom* lAtom = HK_GET_LOCAL_CONTACT_ATOM(m_contactConstraintData.m_atom);
		int numUserDatasA = lAtom->m_numUserDatasForBodyA;
		int numUserDatasB = lAtom->m_numUserDatasForBodyB;

		if ( numUserDatasA + numUserDatasB )
		{
			const hkpCdBody* cdBodyA;
			const hkpCdBody* cdBodyB;
			if ( ea == sortedEa )
			{
				cdBodyA = &a;
				cdBodyB = &b;
			}
			else
			{
				ea = sortedEb;
				cdBodyA = &b;
				cdBodyB = &a;
			}
			
			HK_ASSERT2(0xad8966aa, numUserDatasA + numUserDatasB <= maxNumExtraDataInEvent, "Not enough room for userDatas in TOI point. This assert is ok to ignore, given that you will not try to access the data later....");

			numUserDatasA = hkMath::min2(numUserDatasA, maxNumExtraDataInEvent);
			numUserDatasB = hkMath::min2(numUserDatasB, maxNumExtraDataInEvent - numUserDatasA);

			//
			// Export extended user data
			//
			hkpContactPointProperties::UserData* userDatasA = (hkpContactPointProperties::UserData*)hkAddByteOffset(static_cast<hkpContactPointProperties*>(cpi), sizeof(hkpContactPointProperties));
			hkpContactPointProperties::UserData* userDatasB = (hkpContactPointProperties::UserData*)hkAddByteOffset(static_cast<hkpContactPointProperties*>(cpi), sizeof(hkpContactPointProperties) + numUserDatasA * sizeof(hkpContactPointProperties::UserData));

			{
				
				// Skip artificial capsule shapes added by the gsk cylinder agent.
				if ( ( cdBodyA->getShapeKey() == HK_INVALID_SHAPE_KEY ) && ( cdBodyA->getParent() ) )
				{
					cdBodyA = cdBodyA->getParent();
				}
				// Write the shape keys for body A.
				for ( int udi = 0; cdBodyA && udi < numUserDatasA; udi++, cdBodyA = cdBodyA->getParent())
				{
					userDatasA[udi] = cdBodyA->getShapeKey(); 
				} 
			}
			{
				
				// Skip artificial capsule shapes added by the gsk cylinder agent.
				if ( ( cdBodyB->getShapeKey() == HK_INVALID_SHAPE_KEY ) && ( cdBodyB->getParent() ) )
				{
					cdBodyB = cdBodyB->getParent();
				}
				// Write the shape keys for body B.
				for ( int udi = 0; cdBodyB && udi < numUserDatasB; udi++, cdBodyB = cdBodyB->getParent())
				{ 
					userDatasB[udi] = cdBodyB->getShapeKey(); 
				} 
			}
		}
	}
	return ea;
}



hkpContactMgr::ToiAccept hkpSimpleConstraintContactMgr::addToiImpl( const hkpCdBody& a, const hkpCdBody& b, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& output, hkTime toi, hkContactPoint& cp, const hkpGskCache* gskCache, hkReal& projectedVelocity, hkpContactPointProperties& properties )
{
	hkpRigidBody* ea = setContactPointProperties( a, b, HK_NUM_EXTENDED_USER_DATAS_IN_TOI_EVENT, &properties );
#if !defined(HK_PLATFORM_SPU)
	hkpRigidBody* eb = hkSelectOther( ea, m_constraint.getRigidBodyA(), m_constraint.getRigidBodyB() );
#else
	extern hkpEntity* g_spuCollisionEventsEntityAOnSpu;
	extern hkpEntity* g_spuCollisionEventsEntityBOnSpu;
	hkpEntity* eb = hkSelectOther( (hkpEntity*)ea, g_spuCollisionEventsEntityAOnSpu, g_spuCollisionEventsEntityBOnSpu );
#endif
	hkpToiPointAddedEvent event( this, &input, &output, &a, &b, &cp, gskCache, &properties, toi, projectedVelocity);
	hkFireContactPointAddedCallback( m_world, ea, eb, event );

	if (event.m_status == HK_CONTACT_POINT_REJECT)
	{
		removeToiImpl( *output.m_constraintOwner.val(), properties );
	}
	else
	{
		projectedVelocity = event.m_projectedVelocity;
	}

	return hkpContactMgr::ToiAccept( hkpContactPointAccept(event.m_status) );
}



hkContactPointId hkpSimpleConstraintContactMgr::addContactPointImpl( const hkpCdBody& a, const hkpCdBody& b, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& output, const hkpGskCache* contactCache, hkContactPoint& cp)
{
	hkpSimpleContactConstraintAtom* localAtom = HK_GET_LOCAL_CONTACT_ATOM(m_contactConstraintData.m_atom);

	int size = localAtom->m_numContactPoints;

	if ( size + m_reservedContactPoints >  localAtom->m_maxNumContactPoints - 2 )
	{
		HK_WARN_ONCE(0x11fce585, "Maximum number of ContactPoints reached, dropping points. Note: This message is fired only once and it means that one object has too many triangles in a specific area");
		return HK_INVALID_CONTACT_POINT;
	}

	if ( size == 0 )
	{
		output.m_constraintOwner->addConstraintToCriticalLockedIsland( &m_constraint );
	}

	hkContactPointId id;
	hkpContactPointProperties* cpi;
	{
		hkContactPoint* dcp;
		id = m_contactConstraintData.allocateContactPoint( *output.m_constraintOwner, &dcp, &cpi );
		HK_ON_DEBUG( localAtom = 0 ); // atom might move in memory
		*dcp = cp;
	}

	hkpRigidBody* rba = setContactPointProperties( a, b, 10000, cpi );
#if !defined(HK_PLATFORM_SPU)
	hkpRigidBody* rbb = hkSelectOther( rba, m_constraint.getRigidBodyA(), m_constraint.getRigidBodyB() );
#else
	extern hkpEntity* g_spuCollisionEventsEntityAOnSpu;
	extern hkpEntity* g_spuCollisionEventsEntityBOnSpu;
	hkpRigidBody* rbb = (hkpRigidBody*)hkSelectOther( (hkpEntity*)rba, g_spuCollisionEventsEntityAOnSpu, g_spuCollisionEventsEntityBOnSpu );
#endif

	// set the current penetration depth
	// get the current velocity
	hkSimdReal projectedVel;
	{
		hkVector4 velA;		rba->getPointVelocity( cp.getPosition(), velA );
		hkVector4 velB;		rbb->getPointVelocity( cp.getPosition(), velB );

		hkVector4 deltaVel; deltaVel.setSub( velA, velB );
		projectedVel = deltaVel.dot<3>( cp.getNormal() );
	}

	cpi->m_impulseApplied = 0.0f;
	cpi->setUserData(0);



		//
		// fire all events
		//
	{
		// HVK-SPU combine
		hkpManifoldPointAddedEvent event( id, this, &input, &output, &a,&b, &cp, contactCache, cpi, projectedVel.getReal());
		hkFireContactPointAddedCallback( m_world, rba, rbb, event );

		if ( event.m_status == HK_CONTACT_POINT_REJECT )
		{
			// Note: This will fire the removal event, so all listeners will be correctly informed of the state change.
			removeContactPoint( id, *output.m_constraintOwner );
			return HK_INVALID_CONTACT_POINT;
		}
		m_contactPointCallbackDelay = event.m_nextProcessCallbackDelay;
	}

	

	// check whether we need callbacks
	{
		if ( !input.m_allowToSkipConfirmedCallbacks || cpi->getRestitution() || rba->areContactListenersAdded() || rbb->areContactListenersAdded() )
			
			// || m_world->m_collisionListeners.getSize()
		{
			// flag master for firing contact callbacks
			output.m_constraintOwner->addCallbackRequest( m_contactConstraintData.m_constraint, hkpConstraintAtom::CALLBACK_REQUEST_NEW_CONTACT_POINT );
		}
		else
		{
			
			hkSimdReal sumInvMass = rba->getRigidMotion()->getMassInv() + rbb->getRigidMotion()->getMassInv();
			hkSimdReal mass; mass.setReciprocal( sumInvMass + hkSimdReal::fromFloat(hkReal(1e-10f)) );
			(mass * hkSimdReal::fromFloat(hkReal(-0.2f)) * projectedVel).store<1>(&cpi->m_impulseApplied);
			cpi->m_internalSolverData = 0;
			cpi->m_internalDataA      = 0;
		}
	}

	return id;
}



void hkpSimpleContactConstraintAtomUtil::removeAtAndCopy(hkpSimpleContactConstraintAtom* atom, int index)
{
	HK_ASSERT2(0x724b10b5, index >= 0 && index < atom->m_numContactPoints , "Out of bound error.");
	int numContactPoints = atom->m_numContactPoints-1;
	atom->m_numContactPoints = hkUint16(numContactPoints);

	hkContactPoint*             cp  = &atom->getContactPoints()[index];
	int                         cppStriding = atom->getContactPointPropertiesStriding();
	hkpContactPointPropertiesStream* cpp = hkAddByteOffset( atom->getContactPointPropertiesStream(), HK_HINT_SIZE16(index) * HK_HINT_SIZE16(cppStriding) );

	for (int i = index; i < numContactPoints; i++)
	{
		*(cp)  = *(cp+1);
		hkpContactPointPropertiesStream* cppNext = hkAddByteOffset(cpp, cppStriding);
		hkString::memCpy4(cpp, cppNext, cppStriding >> 2);
		cp++;
		cpp = cppNext;
	}
}


int hkpSimpleContactConstraintData::freeContactPoint( hkpConstraintOwner& constraintOwner, hkContactPointId id )
{
	int mid = int(id);
	HK_ASSERT2(0x7d4661da,  mid>=0, "hkContactConstraint::freeContactPoint(): Contact point was not found in current list");

	hkpSimpleContactConstraintAtom* localAtom = HK_GET_LOCAL_CONTACT_ATOM(m_atom);

	int indexOfRemovePoint = m_idMgrA.getValueAt(mid);
	m_idMgrA.freeId( mid );

	hkpConstraintInfo conInfo;
	conInfo.clear();

#if !defined(HK_PLATFORM_SPU)
	const hkpSimpleContactConstraintAtom* originalAtomPpu = m_atom;
	hkPadSpu<bool> atomReallocated = false;
#endif
	int numContactPoints = localAtom->m_numContactPoints;
	{
		if ( numContactPoints == 2 )
		{
			conInfo.add( hkpJacobianSchemaInfo::Friction3D::Sizeof - hkpJacobianSchemaInfo::Friction2D::Sizeof, 1, (1+0) ); // tempElem for friction doesn't change
		}

		if (numContactPoints & 1) // size before deallocation of an old point
		{
			conInfo.add( hkpJacobianSchemaInfo::SingleContact::Sizeof, 1, 1 );
		}
		else
		{
			conInfo.add( hkpJacobianSchemaInfo::PairContact::Sizeof - hkpJacobianSchemaInfo::SingleContact::Sizeof, 1, 1 );
		}

		hkpSimpleContactConstraintAtomUtil::removeAtAndCopy( localAtom, indexOfRemovePoint );
		{
			hkUchar& flags = const_cast<hkUchar&>((localAtom->getContactPointPropertiesStream(indexOfRemovePoint)->asProperties())->m_flags);
			flags &= ~hkpContactPointProperties::CONTACT_USES_SOLVER_PATH2;	// unchecked overwrite
		}

#if !defined(HK_PLATFORM_SPU)
		m_atom = hkpSimpleContactConstraintAtomUtil::optimizeCapacity(m_atom, 1, atomReallocated);
		localAtom = HK_GET_LOCAL_CONTACT_ATOM(m_atom);
#endif

		m_idMgrA.decrementValuesGreater( indexOfRemovePoint );
	}

	constraintOwner.subConstraintInfo( m_constraint, conInfo );

#if !defined(HK_PLATFORM_SPU)
	if (atomReallocated.val())
	{
		// we know that during this call, no constraints are removed or added by other threads, therefore
		// we can simply modify data which is local to this constraint (hkConstraintInternal)
		hkpWorldConstraintUtil::updateFatherOfMovedAtom( m_constraint, originalAtomPpu, m_atom, m_atom->m_sizeOfAllAtoms );
		m_atomSize = m_atom->m_sizeOfAllAtoms;
	}
#endif
	localAtom->m_info.m_flags |= hkpSimpleContactConstraintDataInfo::HK_FLAG_POINT_REMOVED | hkpSimpleContactConstraintDataInfo::HK_FLAG_AREA_CHANGED;
	return numContactPoints-1;
}


void hkpSimpleConstraintContactMgr::removeContactPointImpl( hkContactPointId cpId, hkCollisionConstraintOwner& constraintOwner )
{
		//
		// fire all events
		//
#if defined(HK_PLATFORM_SPU)
	extern hkpEntity* g_spuCollisionEventsEntityAOnSpu;
	extern hkpEntity* g_spuCollisionEventsEntityBOnSpu;
	hkpEntity* entityA = g_spuCollisionEventsEntityAOnSpu;
	hkpEntity* entityB = g_spuCollisionEventsEntityBOnSpu;
#else
	hkpEntity* entityA = m_constraint.getEntityA();
	hkpEntity* entityB = m_constraint.getEntityB();
#endif
	{
		hkpContactPointProperties* prop = m_contactConstraintData.hkpSimpleContactConstraintData::getContactPointProperties( cpId );
		hkpContactPointRemovedEvent event( cpId, this,  &constraintOwner, prop, entityA, entityB );
		hkFireContactPointRemovedCallback( m_world, entityA, entityB, event );
	}

	int numRemainingContacts = m_contactConstraintData.freeContactPoint( constraintOwner, cpId );

	if( numRemainingContacts == 0 )
	{
		constraintOwner.removeConstraintFromCriticalLockedIsland(&m_constraint);
	}
}

#if !defined(HK_PLATFORM_SPU)
// hkpDynamicsContactMgr interface implementation
hkContactPoint* hkpSimpleConstraintContactMgr::getContactPoint( hkContactPointId id )
{
	hkContactPoint* dcp = &m_contactConstraintData.getContactPoint( id );
	return dcp;
}

hkpContactPointProperties* hkpSimpleConstraintContactMgr::getContactPointProperties( hkContactPointId id )
{
	return m_contactConstraintData.getContactPointProperties( id );
}


void hkpSimpleConstraintContactMgr::getAllContactPointIds( hkArray<hkContactPointId>& contactPointIds ) const
{
	m_contactConstraintData.m_idMgrA.getAllUsedIds(contactPointIds);
}

void hkpSimpleConstraintContactMgr::toiCollisionResponseBeginCallback( const hkContactPoint& cp, struct hkpSimpleConstraintInfoInitInput& inA, struct hkpBodyVelocity& velA, hkpSimpleConstraintInfoInitInput& inB, hkpBodyVelocity& velB)
{
	m_contactConstraintData.collisionResponseBeginCallback( cp, inA, velA, inB, velB );
}



void hkpSimpleConstraintContactMgr::toiCollisionResponseEndCallback( const hkContactPoint& cp, hkReal impulseApplied, struct hkpSimpleConstraintInfoInitInput& inA, struct hkpBodyVelocity& velA, hkpSimpleConstraintInfoInitInput& inB, hkpBodyVelocity& velB)
{
	m_contactConstraintData.collisionResponseEndCallback( cp, impulseApplied, inA, velA, inB, velB );
}
#endif

void hkpSimpleConstraintContactMgr::processContactImpl( const hkpCollidable& a,	const hkpCollidable& b,
													const hkpProcessCollisionInput& input,
													hkpProcessCollisionData& collisionData )
{
#ifdef HK_DEBUG
	int size = collisionData.getNumContactPoints();

	hkpSimpleContactConstraintAtom* localAtom = HK_GET_LOCAL_CONTACT_ATOM(m_contactConstraintData.m_atom);

	HK_ASSERT2(0x25d95ef3, size < localAtom->m_maxNumContactPoints, "Too many contact points in a single collision pair. The system only handles 255 contact points or less between two objects.\
This is probably the result of creating bad collision geometries (i.e. meshes with many triangles in the same place) or having a too large collision tolerance. \
It can also result from not creating a hkpBvTreeShape about your mesh shape.");

	// Check uniqueness of contact points
	if(1)
	{
		HK_ASSERT2(0x38847077,  collisionData.getNumContactPoints() <= localAtom->m_maxNumContactPoints, "Too many contact points between colliding entities" );

		for (int i = 1; i < size; i++ )
		{
			hkpProcessCdPoint& pp = collisionData.m_contactPoints[i];
			HK_ASSERT2( 0xf0f3eaad, hkMath::equal( pp.m_contact.getNormal().lengthSquared<3>().getReal(), 1.0f, 0.01f), "CollisionNormal is not normalized" );
			for (int j = 0; j < i; j++ )
			{
				HK_ASSERT2(0x31ac481b,  pp.m_contactPointId !=  collisionData.m_contactPoints[j].m_contactPointId, "Agent reported a contact point twice" );
			}
		}

	}

	//HK_ASSERT2(0x76cf302f, size == localAtom->m_numContactPoints, "hkpProcessCollisionOutput has a different number of contact points than the hkpContactMgr" );
#endif


	//
	//	fire all events. Do not fire events every frame but use m_skipNextNcallbackSweeps to reduce the frequency
	//
	{
		int skipN = m_contactPointCallbackDelay;
		if ( skipN-- == 0 )
		{
			hkpRigidBody* rba = static_cast<hkpRigidBody*>(a.getOwner());
			hkpRigidBody* rbb = static_cast<hkpRigidBody*>(b.getOwner());

			skipN = hkMath::min2( rba->getContactPointCallbackDelay(), rbb->getContactPointCallbackDelay() );

			//
			// fire all events. 
			//
			{
				hkpContactProcessEvent event( this, &a, &b, &collisionData );

#if !defined(HK_PLATFORM_SPU)
				hkpProcessCdPoint* ccpEnd = collisionData.getEnd();
				hkpContactPointProperties** cppp = &event.m_contactPointProperties[0];
				for( hkpProcessCdPoint* ccp = collisionData.getFirstContactPoint(); ccp < ccpEnd; ccp++)
				{
					*(cppp++) = m_contactConstraintData.getContactPointProperties( ccp->m_contactPointId );
				}
#endif
				hkFireContactProcessCallback( m_world, rba, rbb, event );
			}
			//
			// If skipN is zero, then we also want contactPointCallbacks fired in the next frame, so set the CallbackRequest.
			//
			collisionData.m_constraintOwner->addCallbackRequest( m_contactConstraintData.m_constraint, hkpConstraintAtom::CALLBACK_REQUEST_CONTACT_POINT_CALLBACK );
		}
		m_contactPointCallbackDelay = hkUint16(skipN);
	}

	{
		hkpProcessCdPoint* ccpEnd = collisionData.getEnd();
		for( hkpProcessCdPoint* ccp = collisionData.getFirstContactPoint(); ccp < ccpEnd; ccp++)
		{
			hkContactPoint& cp = m_contactConstraintData.getContactPoint( ccp->m_contactPointId );
			ccp->m_contact.getPosition().store<4,HK_IO_NOT_CACHED>( (hkReal*)&cp.getPosition() );
			ccp->m_contact.getSeparatingNormal().store<4,HK_IO_NOT_CACHED>( (hkReal*)&cp.getSeparatingNormal() );
		}
	}
}


#if !defined(HK_PLATFORM_SPU)
hkpSimpleConstraintContactMgr::Factory::Factory(hkpWorld *mgr)
{
	m_world = mgr;
}

hkpContactMgr*	hkpSimpleConstraintContactMgr::Factory::createContactMgr(const  hkpCollidable& a,const  hkpCollidable& b, const hkpCollisionInput& input )
{
	hkpRigidBody* bodyA = reinterpret_cast<hkpRigidBody*>(a.getOwner() );
	hkpRigidBody* bodyB = reinterpret_cast<hkpRigidBody*>(b.getOwner() );

	hkpSimpleConstraintContactMgr *mgr = new hkpSimpleConstraintContactMgr( m_world, bodyA, bodyB );
	return mgr;
}
#endif

#if defined(HK_REAL_IS_DOUBLE)
	HK_COMPILE_TIME_ASSERT( sizeof(hkContactPoint) == 64 );
#else
	HK_COMPILE_TIME_ASSERT( sizeof(hkContactPoint) == 32 );
#endif
//HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF( hkpSimpleConstraintContactMgr, m_skinNextNprocessCallbacks ) == 12 );

#if HK_POINTER_SIZE == 4
HK_COMPILE_TIME_ASSERT( sizeof(hkpSimpleConstraintContactMgr) == 128 );
#endif

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
