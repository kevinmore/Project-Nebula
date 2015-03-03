/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/CharacterControl/CharacterRigidBody/hkpCharacterRigidBodyListener.h>

#include <Physics2012/Utilities/CharacterControl/CharacterRigidBody/hkpCharacterRigidBody.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnTrack.h>
#include <Physics2012/Dynamics/Collide/hkpResponseModifier.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Dynamics/Collide/hkpSimpleConstraintContactMgr.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>

void hkpCharacterRigidBodyListener::processActualPoints( const hkpWorld* world, hkpCharacterRigidBody* characterRB, const hkpLinkedCollidable::CollisionEntry& entry, hkpSimpleConstraintContactMgr* mgr, hkArray<hkContactPointId>& contactPointIds )
{
	unweldContactPoints( characterRB, entry, mgr, contactPointIds );
	considerCollisionEntryForSlope( world, characterRB, entry, mgr, contactPointIds );
	considerCollisionEntryForMassModification( world, characterRB, entry, mgr, contactPointIds );
}

void hkpCharacterRigidBodyListener::characterCallback( hkpWorld* world, hkpCharacterRigidBody* characterRB )
{
	world->lockCriticalOperations();

	HK_TIMER_BEGIN("characterCallback", HK_NULL);

	discardVerticalPoints( characterRB );
	processActualPoints( world, characterRB );

	HK_TIMER_END();

	world->unlockAndAttemptToExecutePendingOperations();
}

void hkpCharacterRigidBodyListener::considerCollisionEntryForSlope( const hkpWorld* world, hkpCharacterRigidBody* characterRB, const hkpLinkedCollidable::CollisionEntry& entry, hkpSimpleConstraintContactMgr* mgr, hkArray<hkContactPointId>& contactPointIds )
{
	const hkpCollidable *const collidableA = entry.m_agentEntry->getCollidableA();
	const hkpCollidable *const collidableB = entry.m_agentEntry->getCollidableB();

	// Ensure the contact normals we consider are pointing the correct way.
	const hkSimdReal mul = ( hkpGetRigidBody( collidableA ) == characterRB->m_character ) ? hkSimdReal_1 : hkSimdReal_Minus1;

	// We may add new ids, so only iterate over the original ones.
	const int numContactPoints = contactPointIds.getSize();
	for ( int j = 0; j < numContactPoints; ++j )
	{
		const hkContactPointId& id = contactPointIds[j];
		// Skip disabled contact points.
		if ( mgr->getContactPointProperties( id )->m_flags & hkContactPointMaterial::CONTACT_IS_DISABLED )
		{
			continue;
		}

		hkContactPoint *const cp = mgr->getContactPoint( id );

		const hkSimdReal surfaceVerticalComponent = cp->getNormal().dot<3>( characterRB->m_up );
		const hkSimdReal surfaceVerticalComponentFromA = mul * surfaceVerticalComponent;

		if ( ( surfaceVerticalComponentFromA > hkSimdReal::fromFloat(0.01f) ) && ( surfaceVerticalComponentFromA < hkSimdReal::fromFloat(characterRB->m_maxSlopeCosine) ) )
		{
			hkpCharacterRigidBody::VertPointInfo info;
			{
				info.m_mgr = mgr;
				info.m_vertPoint.setPosition( cp->getPosition() );
				hkVector4 cpPos = info.m_vertPoint.getPosition(); cpPos.setInt24W( hkpCharacterRigidBody::m_magicNumber ); info.m_vertPoint.setPosition(cpPos);
				hkVector4 cpN = cp->getSeparatingNormal();
				cpN.addMul( -surfaceVerticalComponent, characterRB->m_up );
				// Normalizing here is safe: m_maxSlopeCosine <= 1, and surfaceVerticalComponentFromA is
				// always positive. Thus, a vertical cp normal wouldn't pass the above test.
				cpN.normalize<3>();
				// Set vertical plane distance to zero unless the character is penetrating the surface.
				hkSimdReal clampedDist; clampedDist.setMin(cp->getDistanceSimdReal(), hkSimdReal_0);
				info.m_vertPoint.setSeparatingNormal(cpN, clampedDist);
			}

			// As it stands, output is only used by hkpSimpleConstraintContactMgr::addContactPointImpl() to obtain
			// a reference to the constraint owner, so we can pass in a local one. The input isn't used at all.
			hkpProcessCollisionOutput output( (hkCollisionConstraintOwner*) characterRB->m_character->getSimulationIsland() );
			hkContactPointId newCpId = mgr->addContactPoint( *collidableA, *collidableB, *world->getCollisionInput(), output, HK_NULL, info.m_vertPoint );

			if (newCpId != HK_INVALID_CONTACT_POINT)
			{
				characterRB->m_verticalContactPoints.pushBack( info );
				contactPointIds.pushBack( newCpId );
			}
		}
	}
}


void hkpCharacterRigidBodyListener::considerCollisionEntryForMassModification( const hkpWorld* world, hkpCharacterRigidBody* characterRB, const hkpLinkedCollidable::CollisionEntry& entry, hkpSimpleConstraintContactMgr* mgr, const hkArray<hkContactPointId>& contactPointIds )
{
	// Ensure the contact normal points in the direction away from character.
	hkpRigidBody* other;
	hkSimdReal mul;
	{
		const hkpCollidable *const collidableA = entry.m_agentEntry->getCollidableA();
		const hkpCollidable *const collidableB = entry.m_agentEntry->getCollidableB();

		if ( hkpGetRigidBody( collidableA ) == characterRB->m_character )
		{
			other = hkpGetRigidBody( collidableB );
			mul = hkSimdReal_Minus1;
		}
		else
		{
			other = hkpGetRigidBody( collidableA );
			mul = hkSimdReal_1;
		}
	}

	// Do only for dynamic bodies.
	if ( ( !other->isFixedOrKeyframed() ) && ( other->getMotionType() !=  hkpMotion::MOTION_CHARACTER ) )
	{
		// Deal with the zero case separately.
		if ( characterRB->m_maxForce > 0.0f )
		{

			// Calculate the sum of all the normals.
			hkVector4 summedNormals;
			{
				summedNormals.setZero();

				const int numContactPoints = contactPointIds.getSize();
				for ( int j = 0; j < numContactPoints; ++j )
				{
					const hkContactPoint *const cp = mgr->getContactPoint( contactPointIds[j] );
					summedNormals.add( cp->getNormal() );
				}

				summedNormals.mul( mul );
			}

			hkSimdReal currentForce; currentForce.setReciprocal<HK_ACC_23_BIT,HK_DIV_SET_ZERO>(characterRB->m_character->getRigidMotion()->getMassInv());
			{
				// HVK-4666: 
				// We'd like to project the acceleration onto the summed normal. In the unlikely event that the
				// normals sum to 0, we just use the length of the acceleration vector instead. This guarantees
				// that we will check for force limiting below.
				hkSimdReal lensq = summedNormals.lengthSquared<3>();

				if ( lensq.isGreaterZero() )
				{
					currentForce.mul( summedNormals.dot<3>( characterRB->m_acceleration ) );
					currentForce.div(lensq);
				}
				else
				{
					currentForce.mul(characterRB->m_acceleration.length<3>());
				}
			}

			hkVector4 factorChar = hkVector4::getConstant<HK_QUADREAL_1>();
			{
				const hkSimdReal charMaxForce = hkSimdReal::fromFloat(characterRB->m_maxForce);
				if ( currentForce > charMaxForce )
				{
					factorChar.setW( currentForce / charMaxForce );
				}
			}

			hkpResponseModifier::setInvMassScalingForContact( mgr, characterRB->m_character, * (hkCollisionConstraintOwner*) characterRB->m_character->getSimulationIsland(), factorChar );
		}
		else
		{
			// Since setInvMassScalingForContact modifies inverse mass, we need a special case
			// when m_maxForce = 0.0f. For the purposes of this collision, make the other entity
			// infinitely massive.

			const hkVector4& factorOther = hkVector4::getConstant<HK_QUADREAL_0>();

			hkpResponseModifier::setInvMassScalingForContact( mgr, other, * (hkCollisionConstraintOwner*) characterRB->m_character->getSimulationIsland(), factorOther );
		}
	}

}


inline hkBool32 pointsMatch ( const hkContactPoint& pointA, const hkContactPoint& pointB )
{
	hkVector4Comparison result  = pointA.getPosition().equal( pointB.getPosition() );
	hkVector4Comparison result2 = pointA.getSeparatingNormal().equal( pointB.getSeparatingNormal() );
	result.setAnd( result, result2 );
	return result.allAreSet();
}


void hkpCharacterRigidBodyListener::discardVerticalPoints( hkpCharacterRigidBody* characterRB )
{
	if ( characterRB->m_verticalContactPoints.getSize() == 0 )
	{
		return;
	}

	hkArray<hkpLinkedCollidable::CollisionEntry> collisionEntriesTmp;
	characterRB->m_character->getLinkedCollidable()->getCollisionEntriesSorted(collisionEntriesTmp);
	const hkArray<hkpLinkedCollidable::CollisionEntry>& collisionEntries = collisionEntriesTmp;
	const int numCollisionEntries = collisionEntries.getSize();

	// We do not attempt to control the lifetime of constraint managers. Thus, we must be very careful when
	// removing contact points, because there's a danger we're looking at a different constraint manager with the
	// same address as the one to which we added points.
	//
	// In particular, we check that contact points have the same magic number in the w component of their position.

	// Remove added points from any extant contact manager.
	for ( int i = 0; i < numCollisionEntries; ++i )
	{
		hkpContactMgr *const cmgr = collisionEntries[i].m_agentEntry->m_contactMgr;
		if ( cmgr->m_type == hkpContactMgr::TYPE_SIMPLE_CONSTRAINT_CONTACT_MGR )
		{
			hkpSimpleConstraintContactMgr *const mgr = static_cast< hkpSimpleConstraintContactMgr* > ( cmgr );

			hkArray<hkContactPointId> contactPointIds;
			mgr->getAllContactPointIds( contactPointIds );

			const int numVertPoints = characterRB->m_verticalContactPoints.getSize();
			for ( int k = numVertPoints - 1; k >= 0; --k )
			{
				const hkpCharacterRigidBody::VertPointInfo& info = characterRB->m_verticalContactPoints[k];
				if ( info.m_mgr == mgr )
				{
					const int numIds = contactPointIds.getSize();
					for ( int j = 0; j < numIds; ++j )
					{
						const hkContactPointId cpId = contactPointIds[j];
						hkContactPoint *const cp = mgr->getContactPoint( cpId );
						if ( pointsMatch( *cp, info.m_vertPoint ) )
						{
							// We ensure the w component of the condemned point no longer matches magic number.
							hkVector4 cpPos = cp->getPosition(); cpPos.setInt24W( hkpCharacterRigidBody::m_notMagicNumber ); cp->setPosition(cpPos);
							mgr->removeContactPoint( cpId, * (hkpConstraintOwner*) characterRB->m_character->getSimulationIsland() );
							characterRB->m_verticalContactPoints.removeAt( k );
							contactPointIds.removeAt( j );
							break;
						}
					}
				}
			}
		}
	}

	// If we have references to contact points which belong to a constraint manager that no longer
	// exists, we needn't worry about them, because they will have been deleted by that contact
	// manager's destructor.
	characterRB->m_verticalContactPoints.clear();
}


void hkpCharacterRigidBodyListener::processActualPoints( hkpWorld* world, hkpCharacterRigidBody* characterRB )
{
	hkArray<hkpLinkedCollidable::CollisionEntry> collisionEntriesTmp;
	characterRB->m_character->getLinkedCollidable()->getCollisionEntriesSorted(collisionEntriesTmp);
	const hkArray<hkpLinkedCollidable::CollisionEntry>& collisionEntries = collisionEntriesTmp;
	const int numCollisionEntries = collisionEntries.getSize();	

	for ( int i = 0; i < numCollisionEntries; ++i )
	{
		const hkpLinkedCollidable::CollisionEntry& entry = collisionEntries[i];

		hkpContactMgr *const cmgr = entry.m_agentEntry->m_contactMgr;

		if ( cmgr->m_type == hkpContactMgr::TYPE_SIMPLE_CONSTRAINT_CONTACT_MGR )
		{
			hkpSimpleConstraintContactMgr *const mgr = static_cast< hkpSimpleConstraintContactMgr* > ( cmgr );

			hkArray<hkContactPointId> contactPointIds;
			mgr->getAllContactPointIds( contactPointIds );
			if( contactPointIds.getSize() )
			{
				processActualPoints( world, characterRB, entry, mgr, contactPointIds );
			}
		}
	}
}


void hkpCharacterRigidBodyListener::unweldContactPoints( hkpCharacterRigidBody* characterRB, const hkpLinkedCollidable::CollisionEntry& entry, hkpSimpleConstraintContactMgr* mgr, const hkArray<hkContactPointId>& contactPointIds )
{
	HK_ASSERT2(0xad3af278, characterRB->m_up.isNormalized<3>() , "characterUp vector should be pre-normalised");

	hkSimdReal normalFactor = entry.m_agentEntry->getCollidableA() == characterRB->m_character->getCollidable() ? hkSimdReal_1 : hkSimdReal_Minus1;
	const hkpShape *const shape = characterRB->m_character->getCollidable()->getShape();

	// Only apply unwelding code to capsules and not when the collision partner is a character.
	if ( shape->getType() == hkcdShapeType::CAPSULE )
	{
		const hkTransform& transform = characterRB->m_character->getTransform();
		const hkpCapsuleShape* capsule = static_cast< const hkpCapsuleShape* >( shape );
		const hkVector4& vertex0 = capsule->getVertex<0>();
		const hkVector4& vertex1 = capsule->getVertex<1>();

		hkVector4 centre, vertexOffsetNormalized;
		hkSimdReal halfSpineLength; hkReal halfUnweldingHeight;
		{
			centre.setAdd( vertex1 , vertex0 );
			centre.mul( hkSimdReal_Inv2 );
			halfSpineLength = centre.distanceTo( vertex0 );
			vertexOffsetNormalized.setSub( vertex0 , centre );
			halfUnweldingHeight = capsule->getRadius() * characterRB->m_unweldingHeightOffsetFactor + hkMath::fabs( vertexOffsetNormalized.dot<3>( characterRB->m_up ).getReal() );
			// Spine length can't be zero because capsules can't have identical vertices
			hkSimdReal invHalfSpineLength; invHalfSpineLength.setReciprocal(halfSpineLength);
			vertexOffsetNormalized.mul(invHalfSpineLength);
		}

		int size = contactPointIds.getSize();
		for ( int i = 0; i < size ; ++i )
		{
			hkContactPointId id = contactPointIds[ i ];
			hkContactPoint* contactPoint = mgr->getContactPoint( id );

			// Do not perform the unwelding code when the contact point is too deeply penetrating.
			if ( ( contactPoint != HK_NULL ) && ( contactPoint->getDistance() > -capsule->getRadius() ) )
			{
				const hkVector4& position = contactPoint->getPosition();

				if ( position.getInt24W() == hkpCharacterRigidBody::m_magicNumber )
				{
					continue;
				}

				hkVector4 positionOffset;
				{
					positionOffset._setTransformedInversePos( transform, position );
					positionOffset.sub( centre );
				}

				if( hkMath::fabs( positionOffset.dot<3>( characterRB->m_up ).getReal() ) < halfUnweldingHeight ) 
				{
					hkVector4 closestPoint;
					{								
						hkSimdReal contactLength = positionOffset.dot<3>( vertexOffsetNormalized );

						if( contactLength > halfSpineLength )
						{
							closestPoint = vertex0;
						}
						else if( contactLength < -halfSpineLength )
						{
							closestPoint = vertex1;
						}
						else
						{
							closestPoint.setMul( contactLength, vertexOffsetNormalized );
							closestPoint.add( centre );					
						}
						closestPoint._setTransformedPos( transform, closestPoint );
					}

					hkVector4 normal;
					{
						normal.setSub( closestPoint , position );
					}

					if ( normal.normalizeIfNotZero<3>() )
					{
						normal.mul( normalFactor );

						contactPoint->setNormalOnly( normal );
					}				
				}
			}
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
