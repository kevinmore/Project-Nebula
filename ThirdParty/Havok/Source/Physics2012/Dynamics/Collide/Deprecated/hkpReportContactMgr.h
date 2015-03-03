/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_DYNAMICS2_REPORT_CONTACT_MGR_H
#define HK_DYNAMICS2_REPORT_CONTACT_MGR_H

#include <Physics2012/Collide/Dispatch/ContactMgr/hkpContactMgrFactory.h>
#include <Physics2012/Dynamics/Collide/hkpDynamicsContactMgr.h>

class hkpWorld;
class hkpCollidable;
struct hkpCollisionInput;

/// Warning: there must be a maximum of ONE instance of this class pointing to a contact point.
/// No other persistent pointers to the contact point are allowed.
/// If the address of this class changes, hkpContactMgr::moveContactPoint() must be called
class hkpReportContactMgr: public hkpDynamicsContactMgr
{
	public:
	HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);
		hkpReportContactMgr( hkpWorld* world, hkpRigidBody *bodyA, hkpRigidBody *bodyB );

		~hkpReportContactMgr();

			/// hkpContactMgr interface implementation.
		hkContactPointId addContactPointImpl( const hkpCdBody& a, const hkpCdBody& b, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& output, const hkpGskCache* contactCache, hkContactPoint& cp );

			/// hkpContactMgr interface implementation.
		hkResult reserveContactPointsImpl( int numPoints ){ return HK_SUCCESS; }

			/// hkpContactMgr interface implementation.
		void removeContactPointImpl( hkContactPointId cpId, hkCollisionConstraintOwner& info );

			/// hkpContactMgr interface implementation.
		void processContactImpl( const hkpCollidable& a, const hkpCollidable& b, const hkpProcessCollisionInput& input, hkpProcessCollisionData& collisionData );

			/// hkpContactMgr interface implementation.
		ToiAccept addToiImpl( const hkpCdBody& a, const hkpCdBody& b, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& output, hkTime toi, hkContactPoint& cp, const hkpGskCache* gskCache, hkReal& projectedVelocity, hkpContactPointProperties& properties );

			/// hkpContactMgr interface implementation.
		void removeToiImpl( class hkCollisionConstraintOwner& constraintOwner, hkpContactPointProperties& properties );

			/// hkpContactMgr interface implementation.
		void confirmToi( struct hkpToiEvent& event, hkReal rotateNormal, class hkArray<class hkpEntity*>& outToBeActivated );

		void cleanup(){ delete this; }

		Type getType() const { return TYPE_REPORT_CONTACT_MGR; }

	public:

			/// Class that creates instances of hkpReportContactMgr.
		class Factory: public hkpContactMgrFactory
		{
			//+vtable(true)
			public:
				HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);
				Factory(hkpWorld* mgr);

				hkpContactMgr*	createContactMgr( const hkpCollidable& a, const hkpCollidable& b, const hkpCollisionInput& input );

			protected:
				hkpWorld* m_world;
		};


	protected:

		hkpRigidBody*				m_bodyA;
		hkpRigidBody*				m_bodyB;
		hkInt16						m_skipNextNprocessCallbacks;
};

#endif // HK_DYNAMICS2_REPORT_CONTACT_MGR_H

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
