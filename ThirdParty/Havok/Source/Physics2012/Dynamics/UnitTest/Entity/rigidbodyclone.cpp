/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

/// Demo framework support
#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

/// Dynamics support
#include <Physics2012/Dynamics/Common/hkpProperty.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>

/// Collide support
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Dispatch/hkpAgentRegisterUtil.h>

/// Utility support
#include <Physics2012/Utilities/Dynamics/Inertia/hkpInertiaTensorComputer.h>

#define HK_RIGID_BODY_CLONE_TEST_INT_PROPERTY_ID 0x1112
#define HK_RIGID_BODY_CLONE_TEST_REAL_PROPERTY_ID 0x1113
#define HK_RIGID_BODY_CLONE_TEST_POINTER_PROPERTY_ID 0x1114

const int   propertyTestInt = 2;
const hkReal propertyTestReal = hkReal(2.2f);
const char* propertyTestString = "propertyTestString\0";

const char* worldObjectName = "worldObjectNameTestString\0";
const int   userData[5] = { 0, 1, 2, 3, 4 };

static hkpRigidBody* createRigidBody( hkpMotion::MotionType motionType )
{
	HK_ASSERT( 0x36a96719, motionType != hkpMotion::MOTION_INVALID && motionType != hkpMotion::MOTION_MAX_ID );

	hkpRigidBody* body = HK_NULL;
	{		
		hkVector4 halfExt; halfExt.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
		hkpBoxShape* boxShape = new hkpBoxShape( halfExt );

		/// Rigid body members
		hkpRigidBodyCinfo info;
		{
			info.m_position.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );			
			hkVector4 axis; axis.set( 0.0f, 1.0f, 0.0f );
			info.m_rotation.setAxisAngle( axis, HK_REAL_PI / ((hkUnitTest::rand01() * 4.0f) + 0.01f) );

			info.m_linearVelocity.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
			info.m_angularVelocity.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
			
			// |[1,1,1]| = 1.732
			info.m_maxAngularVelocity = hkUnitTest::randRange( 1.733f, 100.0f );
			info.m_maxLinearVelocity = hkUnitTest::randRange( 1.733f, 100.0f );

			info.m_linearDamping = hkUnitTest::rand01();
			info.m_angularDamping = hkUnitTest::rand01();

			info.m_friction = hkUnitTest::rand01();
			info.m_restitution = hkUnitTest::rand01();			

			if( motionType == hkpMotion::MOTION_FIXED || motionType == hkpMotion::MOTION_KEYFRAMED )
			{
				info.m_motionType = motionType;								
				info.m_enableDeactivation = (hkUnitTest::rand01() >= 0.5f);
				info.m_qualityType = HK_COLLIDABLE_QUALITY_FIXED;
			}
			else
			{
				info.m_motionType = hkpMotion::MOTION_BOX_INERTIA;
				info.m_enableDeactivation = (hkUnitTest::rand01() >= 0.5f);

				hkMassProperties massProperties;
				{
					hkpInertiaTensorComputer::computeBoxVolumeMassProperties( boxShape->getHalfExtents(), hkUnitTest::rand01(), massProperties );
				}
				info.m_mass = massProperties.m_mass;
				info.m_inertiaTensor = massProperties.m_inertiaTensor;

				info.m_qualityType = static_cast<hkpCollidableQualityType> (hkInt8(hkUnitTest::randRange( HK_COLLIDABLE_QUALITY_DEBRIS, HK_COLLIDABLE_QUALITY_MAX - 1 )));
			}
		}

		/// Entity members
		{
			info.m_collisionResponse = static_cast<hkpMaterial::ResponseType> (hkInt8(hkUnitTest::randRange( hkpMaterial::RESPONSE_INVALID + 1, hkpMaterial::RESPONSE_MAX_ID - 1)));
			info.m_contactPointCallbackDelay = hkInt16(hkUnitTest::randRange( 0, 100 ));
		}
		
		/// World Object members
		hkArray<hkpProperty> properties;
		{
			info.m_collisionFilterInfo = hkInt16(hkUnitTest::randRange( 0, 100 ));
			info.m_shape = boxShape;
		}		

		hkError::getInstance().setEnabled(0x23a78ac2, false);
		body = new hkpRigidBody( info );
		hkError::getInstance().setEnabled(0x23a78ac2, true);
		
		/// Non C-info members
		{
			body->setName( worldObjectName );
			body->setUserData( reinterpret_cast<hkUlong>(userData) );

			body->addProperty( HK_RIGID_BODY_CLONE_TEST_INT_PROPERTY_ID, hkpPropertyValue(propertyTestInt) );
			body->addProperty( HK_RIGID_BODY_CLONE_TEST_REAL_PROPERTY_ID, hkpPropertyValue(propertyTestReal) );
			body->addProperty( HK_RIGID_BODY_CLONE_TEST_POINTER_PROPERTY_ID, hkpPropertyValue( const_cast<char*>(propertyTestString)) );
		}
		
		boxShape->removeReference();
	}

	return body;
}

static void checkHavok230Equality( hkpRigidBody* a, hkpRigidBody* b )
{
	/// Rigid body members
	const hkSimdReal eps = hkSimdReal::fromFloat(1e-3f);
	{
		HK_ASSERT( 0x36a96719, a->getPosition().allEqual<3>( b->getPosition(), eps ) );
		HK_ASSERT( 0x36a96719, a->getRotation().getImag().allEqual<4>( b->getRotation().getImag(), eps ) );
		HK_ASSERT( 0x36a96719, a->getRotation().getReal() == b->getRotation().getReal() );
					
		HK_ASSERT( 0x36a96719, a->getLinearVelocity().allEqual<3>( b->getLinearVelocity(), eps ) );
		HK_ASSERT( 0x36a96719, a->getAngularVelocity().allEqual<3>( b->getAngularVelocity(), eps ) );
		
		HK_ASSERT( 0x36a96719, a->getLinearDamping() == b->getLinearDamping() );
		HK_ASSERT( 0x36a96719, a->getAngularDamping() == b->getAngularDamping() );
		
		HK_ASSERT( 0x36a96719, a->getMaterial().getFriction() == b->getMaterial().getFriction() );
		HK_ASSERT( 0x36a96719, a->getMaterial().getRestitution() == b->getMaterial().getRestitution() );

		HK_ASSERT( 0x36a96719, a->isDeactivationEnabled() == b->isDeactivationEnabled() );

		hkMatrix3 inertiaA;
		hkMatrix3 inertiaB;

		a->getInertiaWorld( inertiaA );
		b->getInertiaWorld( inertiaB );

		HK_ASSERT( 0x36a96719, inertiaA.isApproximatelyEqual( inertiaB ) );

		HK_ON_DEBUG( hkReal massA = a->getMass() );
		HK_ON_DEBUG( hkReal massB = b->getMass() );
		HK_ASSERT( 0x36a96719, hkMath::equal( massA, massB ) );
		HK_ASSERT( 0x36a96719, a->getCenterOfMassLocal().allEqual<3>( b->getCenterOfMassLocal(), eps ) );

	}

	/// Entity members
	{
		HK_ASSERT( 0x36a96719, a->getMaterial().getResponseType() == b->getMaterial().getResponseType() );
			
		HK_ASSERT( 0x36a96719, a->getContactPointCallbackDelay() == b->getContactPointCallbackDelay() );
	}

	/// WorldObject members
	{
		/// Collision related
		HK_ASSERT( 0x36a96719, a->getCollidable()->getCollisionFilterInfo() == b->getCollidable()->getCollisionFilterInfo() );
		HK_ASSERT( 0x36a96719, a->getCollidable()->getShape() == b->getCollidable()->getShape() );
		HK_ASSERT( 0x36a96719, a->getCollidable()->getBroadPhaseHandle()->getType() ==  b->getCollidable()->getBroadPhaseHandle()->getType() );		
		
		/// Name
		HK_ASSERT( 0x36a96719, hkString::strCmp( a->getName(), b->getName() ) == 0 );

		/// User data
		HK_ASSERT( 0x36a96719, a->getUserData() == b->getUserData() );

		/// Properties
		{			
			HK_ASSERT( 0x36a96719, a->getProperty(HK_RIGID_BODY_CLONE_TEST_INT_PROPERTY_ID).getInt() == b->getProperty(HK_RIGID_BODY_CLONE_TEST_INT_PROPERTY_ID).getInt() );
			HK_ASSERT( 0x36a96719, a->getProperty(HK_RIGID_BODY_CLONE_TEST_REAL_PROPERTY_ID).getReal() == b->getProperty(HK_RIGID_BODY_CLONE_TEST_REAL_PROPERTY_ID).getReal() );				
			
			HK_ON_DEBUG(const char* stringA = reinterpret_cast<const char*> (a->getProperty(HK_RIGID_BODY_CLONE_TEST_POINTER_PROPERTY_ID).getPtr()));
			HK_ON_DEBUG(const char* stringB = reinterpret_cast<const char*> (b->getProperty(HK_RIGID_BODY_CLONE_TEST_POINTER_PROPERTY_ID).getPtr()));
			
			HK_ASSERT( 0x36a96719, hkString::strCmp( stringA, stringB ) == 0 );
		}		

	}
}

static void checkHavok300Equality( hkpRigidBody* a, hkpRigidBody* b )
{
	/// Rigid Body Members
	{
		HK_ASSERT( 0x36a96719, a->getMaxAngularVelocity() == b->getMaxAngularVelocity() );
		HK_ASSERT( 0x36a96719, a->getMaxLinearVelocity() == b->getMaxLinearVelocity() );
	}
	/// World Object members
	{
		HK_ASSERT( 0x36a96719 , a->getCollidable()->getBroadPhaseHandle()->m_objectQualityType ==  b->getCollidable()->getBroadPhaseHandle()->m_objectQualityType );		
		HK_ASSERT( 0x36a96719, a->getCollidable()->getAllowedPenetrationDepth() == b->getCollidable()->getAllowedPenetrationDepth() );					
	}
}

static void checkUserAllocatedData( hkpRigidBody* body )
{
	/// Check properties
	{		
		HK_ASSERT( 0x36a96719, body->getProperty( HK_RIGID_BODY_CLONE_TEST_INT_PROPERTY_ID ).getInt() == propertyTestInt );
		HK_ASSERT( 0x36a96719, body->getProperty( HK_RIGID_BODY_CLONE_TEST_REAL_PROPERTY_ID ).getReal() == propertyTestReal );

		HK_ON_DEBUG( const char* pointerProperty = reinterpret_cast<const char*> (body->getProperty(HK_RIGID_BODY_CLONE_TEST_POINTER_PROPERTY_ID).getPtr()) );

		HK_ASSERT( 0x36a96719, hkString::strCmp( pointerProperty, propertyTestString ) == 0 );

		HK_ASSERT( 0x36a96719, pointerProperty == propertyTestString  );
	}	
}

static void checkRigidBody( hkpRigidBody* body )
{
	hkError::getInstance().setEnabled(0x23a78ac2, false);
	hkpRigidBody* cloneOfBody = body->clone();
	hkError::getInstance().setEnabled(0x23a78ac2, true);
		
	checkHavok230Equality( body, cloneOfBody );
	checkHavok300Equality( body, cloneOfBody );
	
	body->removeReference();

	checkUserAllocatedData( cloneOfBody );

	cloneOfBody->removeReference();
}

// This test the clone funtionality of hkpRigidBody::clone.  To pass this test a rigid body must clone
// all of the persistent properties maintained in Havok 2.3.1 hkpRigidBodyCinfo as well as noted
// Havok 3 additions.  These are
//
//	Havok 2.3.1 
//
// 	-hkpRigidBodyCinfo			
//		m_position
//		m_rotation
//		m_linearVelocity
//		m_angularVelocity
//		m_inertiaTensor	
//		m_centerOfMass	
//		m_mass			
//		m_linearDamping	
//		m_angularDamping
//		m_friction		
//		m_restitution	
//		m_motionType	
//		m_rigidBodyActivatorType
//	-hkEntityCinfo	
//		m_collisionResponse
//		m_processContactCallbackDelay					
//	-hkWorldObjectCinfo
//		m_collisionFilterInfo								
//		m_shape									
//		m_broadPhaseType						
//		m_properties
//      m_name
//      m_userData
//
//	Havok 3.0.0	
//
//	-hkpRigidBody
//		m_maxLinearVelocity
//		m_maxAngularVelocity
//      m_allowedPenetrationDepth
//	-hkpEntity
//		none (could clone listeners?)
//	-hkpWorldObject
//		m_objectQualityType 

static int rigidbodyclone_main()
{
	// Test in non world
	{
		// Fixed body	
		{						
			hkpRigidBody* fixedBody = createRigidBody( hkpMotion::MOTION_FIXED );

			checkRigidBody( fixedBody );			
		}	

		// Keyframed body
		{
			hkpRigidBody* keyFramedBody = createRigidBody( hkpMotion::MOTION_KEYFRAMED );

			checkRigidBody( keyFramedBody );					
		}

		// Dynamic body
		{
			hkpRigidBody* dynamicBody = createRigidBody( hkpMotion::MOTION_BOX_INERTIA );
			
			checkRigidBody( dynamicBody );							
		}
	}

	// Test in world
	{
		hkpWorldCinfo info;
		hkpWorld* world = new hkpWorld( info );
		world->lock();
		
		hkpAgentRegisterUtil::registerAllAgents(world->getCollisionDispatcher());

		// Fixed body	
		{
			hkpRigidBody* fixedBody = createRigidBody( hkpMotion::MOTION_FIXED );

			world->addEntity( fixedBody )->removeReference();			
		}	

		// Keyframed body
		{
			hkpRigidBody* keyFramedBody = createRigidBody( hkpMotion::MOTION_KEYFRAMED );

			world->addEntity( keyFramedBody )->removeReference();			
		}

		// Dynamic body
		{
			hkpRigidBody* dynamicBody = createRigidBody( hkpMotion::MOTION_BOX_INERTIA );
			
			world->addEntity( dynamicBody )->removeReference();			
		}

		world->unlock();
		world->removeReference();
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(rigidbodyclone_main, "Fast", "Physics2012/Test/UnitTest/Dynamics/", __FILE__     );

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
