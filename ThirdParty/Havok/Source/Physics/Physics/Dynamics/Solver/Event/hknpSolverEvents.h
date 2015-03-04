/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SOLVER_EVENTS_H
#define HKNP_SOLVER_EVENTS_H

#include <Common/Base/Container/CommandStream/hkCommandStream.h>
#include <Geometry/Internal/Types/hkcdManifold4.h>

#include <Physics/Physics/Dynamics/World/Events/hknpEvents.h>
#include <Physics/Physics/Dynamics/Modifier/hknpModifier.h>
#include <Physics/Physics/Dynamics/Constraint/hknpConstraint.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobian.h>

class hknpWorld;


/// A solver event for a single manifold of a contact Jacobian.
struct hknpContactSolverEvent : public hknpBinaryBodyEvent
{
	public:

		/// Constructor.
		hknpContactSolverEvent( hkUint16 subType, int sizeInBytes, hknpBodyId idA, hknpBodyId idB ) :
			hknpBinaryBodyEvent( subType, sizeInBytes, idA, idB ),
			m_contactJacobian( HK_NULL ),
			m_manifoldIndex(0)
		{}

		/// Initialize from a modifier callback input.
		HK_FORCE_INLINE void initialize( const hknpModifier::SolverCallbackInput& input );

#if !defined(HK_PLATFORM_SPU)

		/// Get the header data from the Jacobian.
		HK_FORCE_INLINE const hknpContactJacobianTypes::HeaderData& getJacobianHeader() const;

		/// Get the number of contact points in the manifold.
		HK_FORCE_INLINE int getNumContactPoints() const;

		/// Calculate pre-integration contact point positions from the Jacobian.
		void calculateContactPointPositions( hknpWorld* world, hkcdManifold4* manifoldOut ) const;

		/// Predict the contact point distances after the integration step.
		void predictCurrentContactPointDistances(
			hknpWorld* world, const hkcdManifold4& manifold, hkVector4* contactPointDistances ) const;

		/// Calculate a value between 0 and 1 which will bring the object in a non penetrating state assuming that
		/// the object was not penetrating the last frame and that the predicted current contact point positions
		/// indicate penetration. This typically happens if the impulses are clipped.
		hkSimdReal calculateToi( hknpWorld* world, hkSimdRealParameter maxExtraPenetration ) const;

#endif

	public:

		/// The contact Jacobian.
		const hknpMxContactJacobian* m_contactJacobian;

		/// Index of the relevant manifold in the contact Jacobian.
		hkUint8 m_manifoldIndex;
};


/// An event for reporting when a manifold starts/continues/finishes applying non-zero impulses in the solver.
struct hknpContactImpulseEvent : public hknpContactSolverEvent
{
	public:

		// Collision status.
		enum Status
		{
			STATUS_NONE			= 0,	///< No collision.
			STATUS_STARTED		= 1,	///< The manifold went from not applying an impulse to applying an impulse
			STATUS_FINISHED		= 2,	///< The manifold went from applying an impulse to not applying an impulse.
			STATUS_CONTINUED	= 3		///< The manifold continued applying an impulse.
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpContactImpulseEvent );

		/// Constructor.
		HK_FORCE_INLINE hknpContactImpulseEvent( hknpBodyId idA, hknpBodyId idB ) :
			hknpContactSolverEvent( hknpEventType::CONTACT_IMPULSE, sizeof(*this), idA, idB ) {}

		/// Set whether to raise further events while the manifold continues to apply impulses.
		HK_FORCE_INLINE void setContinuedEventsEnabled( bool areEnabled ) const;

		/// Set whether to raise an event when the manifold has finished applying impulses.
		/// Note: The manifold data may not be available for these events, as the manifold may have been destroyed.
		HK_FORCE_INLINE void setFinishedEventsEnabled( bool areEnabled ) const;

		/// Print this.
		void printCommand( hknpWorld* world, hkOstream& stream ) const;

	public:

		/// The collision status.
		hkEnum<Status, hkUint8> m_status;

		/// A value between 0 and 1 describing how much friction force has been applied relative to the theoretical
		/// friction force for infinite friction.
		///   0: The bodies slid against each other
		///   1: The bodies did not slide
		hkReal m_frictionFactor;

		/// The impulse applied to each contact point in the manifold.
		HK_ALIGN_REAL( hkReal m_contactImpulses[4] );
};

HKNP_DECLARE_EVENT_DISCRIMINATOR( hknpContactImpulseEvent, hknpEventType::CONTACT_IMPULSE );


/// An event for reporting when contact impulses were clipped to their Jacobian's maximum impulse value.
struct hknpContactImpulseClippedEvent : public hknpContactSolverEvent
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpContactImpulseClippedEvent );

		hknpContactImpulseClippedEvent( hknpBodyId idA, hknpBodyId idB ) :
			hknpContactSolverEvent( hknpEventType::CONTACT_IMPULSE_CLIPPED, sizeof(*this), idA, idB ) {}

		void printCommand( hknpWorld* world, hkOstream& stream ) const;

	public:

		/// Sum of the unclipped impulses.
		hkReal m_sumContactImpulseUnclipped;
};

HKNP_DECLARE_EVENT_DISCRIMINATOR( hknpContactImpulseClippedEvent, hknpEventType::CONTACT_IMPULSE_CLIPPED );


/// Constraint force events.
struct hknpConstraintForceEvent : public hknpBinaryBodyEvent
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpConstraintForceEvent );

	/// Constructor.
	hknpConstraintForceEvent( hknpConstraint* constraint ) :
		hknpBinaryBodyEvent( hknpEventType::CONSTRAINT_FORCE, sizeof(*this), constraint->m_bodyIdA, constraint->m_bodyIdB ),
		m_constraint(constraint) {}

	/// Print.
	void printCommand( hknpWorld* world, hkOstream& stream ) const;

	/// Constraint instance.
	hknpConstraint* m_constraint;
};

HKNP_DECLARE_EVENT_DISCRIMINATOR( hknpConstraintForceEvent, hknpEventType::CONSTRAINT_FORCE );


/// An event fired when constraint forces are exceeded.
struct hknpConstraintForceExceededEvent : public hknpBinaryBodyEvent
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpConstraintForceExceededEvent );

	/// Constructor.
	hknpConstraintForceExceededEvent( hknpConstraint* constraint ) :
		hknpBinaryBodyEvent( hknpEventType::CONSTRAINT_FORCE_EXCEEDED, sizeof(*this), constraint->m_bodyIdA, constraint->m_bodyIdB ),
		m_constraint(constraint) {}

	/// Print.
	void printCommand( hknpWorld* world, hkOstream& stream ) const;

	/// Constraint instance.
	hknpConstraint* m_constraint;
};

HKNP_DECLARE_EVENT_DISCRIMINATOR( hknpConstraintForceExceededEvent, hknpEventType::CONSTRAINT_FORCE_EXCEEDED );


/// An event fired if the body did not receive the full integration because
/// hknpMotion::m_maxLinearAccelerationDistancePerStep was set.
struct hknpLinearIntegrationClippedEvent : public hknpUnaryBodyEvent
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpLinearIntegrationClippedEvent );

	/// Constructor.
	hknpLinearIntegrationClippedEvent( hknpBodyId id, hkVector4Parameter stolenVelocity ) :
		hknpUnaryBodyEvent( hknpEventType::LINEAR_INTEGRATION_CLIPPED, sizeof(*this), id ),
		m_stolenVelocity( stolenVelocity ) {}

	/// Print.
	void printCommand( hknpWorld* world, hkOstream& stream ) const;

	/// The velocity part which was not taken into account when the rigid body was integrated.
	hkVector4 m_stolenVelocity;
};

HKNP_DECLARE_EVENT_DISCRIMINATOR( hknpLinearIntegrationClippedEvent, hknpEventType::LINEAR_INTEGRATION_CLIPPED );


#include <Physics/Physics/Dynamics/Solver/Event/hknpSolverEvents.inl>


#endif	// HKNP_SOLVER_EVENTS_H

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
