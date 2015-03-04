/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CHARACTER_RIGID_BODY_H
#define HKNP_CHARACTER_RIGID_BODY_H

// Enable this to see manifold planes.
#ifdef HK_DEBUG
// Uncomment this to show debug information
//#define DEBUG_HKNP_CHARACTER_RIGIDBODY
#endif

#include <Physics/Physics/hknpTypes.h>

#include <Physics/Physics/Extensions/CharacterControl/RigidBody/hknpCharacterRigidBodyCinfo.h>
#include <Physics/Physics/Extensions/CharacterControl/hknpCharacterSurfaceInfo.h>

#include <Common/Base/Types/Physics/ContactPoint/hkContactPoint.h>
#include <Geometry/Internal/Types/hkcdManifold4.h>

class hkStepInfo;
class hknpCharacterRigidBodyListener;
class hkContactPoint;
class hknpSolverData;
struct hknpManifoldProcessedEvent;


/// Wrapper around a rigid body to treat it as a character controller.
class hknpCharacterRigidBody : public hkReferencedObject
{
	public:

		/// A structure describing a supporting surface and the contact point between it and the character.
		/// The contact point normal always points towards the character.
		struct SupportInfo
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCharacterRigidBody::SupportInfo );

			hkContactPoint m_point;
			hknpBodyId m_rigidBody;
		};

		/// Supported contact point classification types.
		enum ContactType
		{
			UNCLASSIFIED = 0x00,	// Invalid
			HORIZONTAL,
			VERTICAL,
			MODIFIED,
			IGNORED,				// contact point will be disabled/not added
		};

		/// The collected and filtered contact point information.
		struct ContactPointInfo
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCharacterRigidBody::ContactPointInfo );
			ContactPointInfo() { m_contactType = UNCLASSIFIED; m_bodyIds[0] = m_bodyIds[1] = hknpBodyId::invalid(); }

			hkEnum<ContactType,hkUint8>	m_contactType;
			hknpBodyId m_bodyIds[2];		///< Body A (this/character) and B (can be 0)
			hkContactPoint m_contactPoint;
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Constructor. Creates the rigid body and a listener.
		hknpCharacterRigidBody( const struct hknpCharacterRigidBodyCinfo& info );

		/// Destructor.
		virtual ~hknpCharacterRigidBody();

		/// Manifold event handler.
		void onManifoldEvent( const hknpEventHandlerInput& input, const hknpEvent& event );

		/// Add a character listener.
		void addListener( hknpCharacterRigidBodyListener* listener );

		/// Remove a hknpCharacterProxyListener.
		void removeListener(hknpCharacterRigidBodyListener* listener);

		/// Get character rigid body ID.
		HK_FORCE_INLINE hknpBodyId getBodyId() const;

		/// Get simulated rigid body character.
		HK_FORCE_INLINE const hknpBody& getSimulatedBody() const;

		/// Check to see if the character is supported from below.
		/// This call checks the geometry immediately around the character, and takes neither velocity nor friction
		/// into account. For example, if the character is moving away from the ground, but is still touching it,
		/// this function will return SUPPORTED as the supported state.
		virtual void checkSupport( const hkStepInfo& stepInfo, hknpCharacterSurfaceInfo& ground ) const;

		/// Determines whether the character is supported and, if so, provides the supporting surfaces.
		/// \param stepInfo the step info of
		/// \param supportInfo an array into which support information will be placed, if the character is supported or sliding.
		virtual hknpCharacterSurfaceInfo::SupportedState getSupportInfo( const hkStepInfo& stepInfo, hkArray<SupportInfo>& supportInfo ) const;

		/// Calculates an "average" surface given an array of support information.
		/// This method does not modify the supportedState member of ground.
		/// If at least one of surfaces is dynamic, the ground is regarded as dynamic.
		/// \param ground the structure into which the surface information is placed.
		/// \param useDynamicBodyVelocities Do the surface velocities of dynamic bodies contribute to the resulting surface?
		virtual void getGround( const hkArray<SupportInfo>& supportInfo, bool useDynamicBodyVelocities, hknpCharacterSurfaceInfo& ground ) const;

		/// Set linear acceleration for mass modifier.
		void setLinearAccelerationToMassModifier(hkVector4Parameter newAcc);

		//
		// Simple wrappers for rigid body functions
		//

		/// Returns true if the body is active.
		HK_FORCE_INLINE hkBool isActive() const;

		/// Set character linear velocity. The timestep argument should be given the same timestep that is passed
		/// to the physics simulation step. Note, the timestep must be greater than 0.0f. A value of 0.0f or less
		/// will set m_acceleration to be INF, which could cause crashes.
		/// Set forceActivate to make sure the rigid body is awake the next frame if it is deactivated.
		void setLinearVelocity(hkVector4Parameter newVel, hkSimdRealParameter timestep, hkBool forceActivate = false);

		/// Get character linear velocity.
		HK_FORCE_INLINE const hkVector4& getLinearVelocity() const;

		/// Set character angular velocity.
		void setAngularVelocity(hkVector4Parameter newAVel);

		/// Get character angular velocity.
		HK_FORCE_INLINE void getAngularVelocity(hkVector4& angVel) const;

		/// Get character position.
		HK_FORCE_INLINE const hkVector4& getPosition() const;

		/// Get character transform.
		HK_FORCE_INLINE const hkTransform& getTransform() const;

		void setPosition(hkVector4Parameter position);
		void setTransform(const hkTransform& transform);

		//
		// Signal handlers
		//

		void onPreCollideSignal( hknpWorld* world );
		void onPostCollideSignal( hknpWorld* world, hknpSolverData* collideOut);
		void onPostSolveSignal( hknpWorld* world );

#ifdef DEBUG_HKNP_CHARACTER_RIGIDBODY
		/// We set the w component of the position of vertical contact points to this value.
		static const int m_magicNumber = 0x008babe6;
		static void HK_CALL debugContactPoint( hkContactPoint& cp, bool isSupportingPoint = false );
		static void HK_CALL debugGround(hkVector4Parameter position, const hknpCharacterSurfaceInfo& ground);
#endif

	public:

		hknpBodyId m_bodyId;

		const hknpShape* m_shape;

		hknpWorld* m_world;

		hkArray<hknpCharacterRigidBodyListener*> m_listeners;

		hkVector4 m_up;

		hkReal m_maxSlopeCosine;

		hkReal m_maxSpeedForSimplexSolver;

		/// A character is considered supported if it is less than this distance from its supporting planes.
		/// This should be less than or equal to the world's collision tolerance.
		hkReal m_supportDistance;

		/// A character will keep falling if it is greater than this distance from its supporting planes.
		/// This should be less than or equal to m_supportDistance.
		hkReal m_hardSupportDistance;

		hkVector4 m_acceleration;

		hkReal m_maxForce;

		/// The filtered contact points we added from the event queue.
		hkArray<ContactPointInfo> m_filteredContactPoints;
};

#include <Physics/Physics/Extensions/CharacterControl/RigidBody/hknpCharacterRigidBody.inl>

#endif // HKNP_CHARACTER_RIGID_BODY_H

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
