/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CHARACTER_PROXY_H
#define HKNP_CHARACTER_PROXY_H

#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Common/Base/Types/Physics/hkStepInfo.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Collide/Query/hknpCollisionQuery.h>
#include <Physics/Physics/Collide/Query/hknpCollisionQueryDispatcher.h>
#include <Physics/Physics/Collide/Event/hknpCollideEvents.h>
#include <Physics/Physics/Extensions/CharacterControl/Proxy/hknpCharacterProxyCinfo.h>

class hknpShape;
class hknpWorld;
class hknpAllHitsCollector;
struct hkSurfaceConstraintInfo;
class hknpCharacterProxyListener;
class hkContactPoint;
struct hknpCharacterSurfaceInfo;
struct hknpCharacterObjectInteractionEvent;
struct hknpCharacterObjectInteractionResult;
struct hknpCollisionResult;

#if defined(HK_DEBUG)
#	define HKNP_DEBUG_CHARACTER_PROXY	// display debugging information
#endif


/// A body property used to identify hknpCharacterProxy bodies.
struct hknpCharacterProxyProperty
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCharacterProxyProperty );
	enum { Key = hknpBodyPropertyKeys::CHARACTER_PROXY };
	class hknpCharacterProxy* m_proxyController;
};


/// The character proxy class is used to represent a non penetrating shape that can move dynamically around
/// the scene. It is called character proxy because it is usually used to represent game characters. It could
/// just as easily be used to represent any dynamic game object. It uses a rigid body that ignores collisions.
class hknpCharacterProxy : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Constructor.
		hknpCharacterProxy(const hknpCharacterProxyCinfo& info);

		/// Destructor.
		virtual ~hknpCharacterProxy();

		/// Add a hknpCharacterProxyListener.
		void addListener(hknpCharacterProxyListener* listener);

		/// Remove a hknpCharacterProxyListener.
		void removeListener(hknpCharacterProxyListener* listener);

		/// Check and see if the character is supported in the given direction (and by what).
		/// This call checks the geometry immediately around the character, and does not take velocity into account.
		/// i.e., if the character is moving away from the ground, but is still touching the ground, this function
		/// will return SUPPORTED as the supported state.
		void checkSupport(hkVector4Parameter direction, hknpCharacterSurfaceInfo& ground);

		/// This function is the same as the checkSupport() function, except it allows you to pass your own collectors to filter
		/// collisions. You only need to call this if you are using the integrateWithCollectors call.
		void checkSupportWithCollector(hkVector4Parameter direction, hknpCharacterSurfaceInfo& ground, hknpAllHitsCollector& startPointCollector);

		/// Update and move the character. To override the logic defining which bodies the character can collide
		/// with or capture this information for your own use (e.g., to deal with other proxies), use ::integrateWithCollectors() instead.
		/// By default ::integrate() uses hknpCdQueryAllHitsCollectors for the character's collision queries.
		/// The worldGravity parameter is only used when the character's mass is greater than 0, to apply downward impulses to dynamic objects.
		void integrate(const hkStepInfo& stepInfo, hkVector4Parameter worldGravity);

		/// Update and move the character. You must pass the two collectors used internally in the linear cast
		/// calls. Implementing your own hknpCdQueryAllHitsCollectors will allow you to customize the character's movement behavior
		/// or extract information about objects it overlaps with or objects it linear casts through.
		/// The gravity parameter is only used when the character's mass is greater than 0, to apply downward impulses to dynamic objects.
		void integrateWithCollectors(const hkStepInfo& stepInfo, hkVector4Parameter gravity, hknpAllHitsCollector& castCollector, hknpAllHitsCollector& startPointCollector);

		/// When hknpCharacterProxy::integrate() is called, or hknpWorld::stepSolve() is called, the manifold
		/// information will become out of date. If you use the manifold in your state machine, and need it to be
		/// up to date, you should call this function first. Note that this function is called by checkSupport automatically
		/// when m_refreshManifoldInCheckSupport is set so in this case if you call checkSupport just before your state
		/// machine you do not need to call this function.
		void refreshManifold(hknpAllHitsCollector& startPointCollector);

		//
		// Accessor methods
		//

		/// Read access to the current manifold for the character.
		HK_FORCE_INLINE const hkArray<hknpCollisionResult>& getManifold() const;

		/// Get current position.
		HK_FORCE_INLINE const hkVector4& getPosition() const;

		/// Warp the character (through walls etc.).
		HK_FORCE_INLINE void setPosition(hkVector4Parameter position);

		/// Gets the linear velocity.
		HK_FORCE_INLINE const hkVector4& getLinearVelocity() const;

		/// Sets the velocity.
		HK_FORCE_INLINE void setLinearVelocity(hkVector4Parameter vel);

		/// Gets the transform.
		HK_FORCE_INLINE const hkTransform& getTransform() const;

		/// Gets the transformed AABB.
		HK_FORCE_INLINE void getAabb(hkAabb& aabb) const;

		/// Sets the character proxy shape, recalculates the cached aabb and invalidates the current manifold.
		void setShape(const hknpShape* shape);

		/// Returns the physics world.
		HK_FORCE_INLINE const hknpWorld* getWorld() const;

		/// Returns the character proxy shape.
		HK_FORCE_INLINE const hknpShape* getShape() const;

		/// Returns the character phantom (will be invalid if presenseInWorld was false upon creation).
		HK_FORCE_INLINE hknpBodyId getPhantom() const;

		/// Returns the maximum slope the character can walk up or stand in (see hknpCharacterProxyCinfo::m_maxSlope).
		HK_FORCE_INLINE hkReal getMaxSlope() const;

		/// Sets the maximum slope the character can walk up or stand in (see hknpCharacterProxyCinfo::m_maxSlope).
		/// Call before checkSupport() for the change to take full effect in the next simulation step.
		HK_FORCE_INLINE void setMaxSlope(hkReal maxSlope);

		/// Signal handler.
		void onPostSolveSignal(hknpWorld* world);

	protected:

		/// Some constants.
		enum
		{
			/// Maximum number of shape keys expected from an AABB query in a composite shape.
			NP_CHARACTER_PROXY_MAX_SHAPE_KEYS = 128,

			/// Maximum number of iterations to be performed during shape casts.
			NP_CHARACTER_PROXY_SHAPE_CAST_ITERATIONS = 8
		};

		/// Used to keep track of overlapping trigger volumes
		struct TriggerVolume
		{
			TriggerVolume() {}

			TriggerVolume(hknpBodyId bodyId, hknpShapeKey shapeKey)
				: m_bodyId(bodyId), m_shapeKey(shapeKey), m_isOverlapping(true) {}

			HK_FORCE_INLINE bool operator==(const TriggerVolume& b)
			{
				return (m_bodyId == b.m_bodyId) && (m_shapeKey == b.m_shapeKey);
			}

			hknpBodyId m_bodyId;
			hknpShapeKey m_shapeKey;
			hkBool m_isOverlapping;
		};

		/// Returned by worldLinearCast when a trigger volume is hit.
		struct TriggerVolumeHit
		{
			HK_DECLARE_PLACEMENT_ALLOCATOR();

			TriggerVolumeHit() {}

			TriggerVolumeHit(hknpBodyId bodyId, hknpShapeKey shapeKey, hkReal fraction)
				: m_trigger(bodyId, shapeKey), m_fraction(fraction) {}

			TriggerVolume m_trigger;
			hkReal m_fraction;
		};

		/// Structure used to hold shape information in shape cast queries.
		struct ShapeInfo
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, ShapeInfo );

			hkUint64 m_userData;
			const hknpBody* m_body;
			const hknpShape* m_shape;
			const hkTransform* m_transform;
			hkUint32 m_collisionFilterInfo;
			hknpShapeKey m_shapeKey;
			hknpMaterialId m_materialId;
		};

		//
		// Queries
		//

		/// Performs a world linear cast of the proxy shape from the current position using the given displacement.
		/// The start collector and trigger hit array are optional.
		void worldLinearCast(hkVector4Parameter displacement, hknpAllHitsCollector& castCollector,
			hknpAllHitsCollector* startCollector, hkLocalArray<TriggerVolumeHit>* triggerHits = HK_NULL) const;

		/// Casts the proxy shape against a leaf (i.e non-composite) target shape. When a start collector is provided hits
		/// within the start point tolerance will be added to it and to the cast collector. The manifold update logic
		/// relies on start hits being present in the cast collector too.
		void shapeCast(
			hkVector4Parameter castPath, hknpAllHitsCollector& castCollector, hknpAllHitsCollector* startCollector,
			hkLocalArray<TriggerVolumeHit>* triggerHits, hknpCollisionQueryContext* queryContext,
			const ShapeInfo& targetShapeInfo, hkSimdRealParameter startPointTolerance) const;

		//
		// Manifold handling
		//

		/// Updates the current manifold with the contact points contained in both collectors in the following way:
		/// 1. Update current manifold contacts with the start collector information.
		///    A contact point is removed from the manifold if:
		///    a) it is no longer valid in the world, or
		///    b) it is not found in the start collector.
		/// 2. Add to the manifold the closest contact in the start collector that is not already in the manifold.
		/// 3. Add to the manifold the closest contact in the cast collector if it is not already in the manifold.
		/// 4. Remove duplicated contacts from the manifold.
		void updateManifold(const hknpAllHitsCollector& startCollector, const hknpAllHitsCollector& castCollector);

		/// Removes all contacts from stored manifold whose bodies have been removed from the world.
		/// Note: this method must be called every frame to guarantee that invalid bodies are caught before their body ids are reused.
		void validateManifold();

		/// Creates the surface constraints that will be used by the simplex solver to calculate the allowed movement for the
		/// character. The character is regarded as a single point in the origin and the constraints are defined by their
		/// normal and distance to the origin (in the normal direction).
		void createConstraintsFromManifold(hkReal timeTravelled, hkArray<hkSurfaceConstraintInfo>& constraints) const;

		/// Creates a single surface constraint from a contact point.
		void createSurfaceConstraint(const hknpCollisionResult& contact, hkReal timeTravelled, hkSurfaceConstraintInfo& constraintOut) const;

		/// Creates a new surface constraint to impede the movement of the character up the surface defined by the given
		/// constraint if the slope of that surface is above the maximum slope.
		static hkBool addMaxSlopePlane(hkReal maxSlopeCos, hkVector4Parameter up, int index, hkArray<hkSurfaceConstraintInfo>& constraints);

		/// Returns a similarity value for the pair of contact points, the closer to zero the more similar. The metric is based
		/// on the angle between normals, the contact point distance and the point velocities.
		hkReal compareContacts(const hknpCollisionResult& contactA, const hknpCollisionResult& contactB) const;

		/// Returns the index of the best match of a contact in the given contact list if the similarity metric value is below
		/// the fitness threshold. Otherwise returns -1.
		int findContact(const hknpCollisionResult& contact, const hkArray<hknpCollisionResult>& contactList, hkReal fitnessThreshold) const;

		/// Calculates and applies collision response impulses to the dynamic bodies the character is in contact with
		/// according to the current manifold.
		void resolveContacts(const hkStepInfo& stepInfo, hkVector4Parameter worldGravity);

		//
		// Phantom
		//

		void addPhantom();

		void removePhantom();

		void updatePhantom();

		//
		// Event handling
		//

		void fireContactAdded(const hknpCollisionResult& point) const;

		void fireContactRemoved(const hknpCollisionResult& point) const;

		void fireConstraintsProcessed(const hkArray<hknpCollisionResult>& manifold, struct hkSimplexSolverInput& input) const;

		void fireCharacterInteraction(hknpCharacterProxy* otherProxy, const hkContactPoint& contact);

		void fireObjectInteraction(const hknpCharacterObjectInteractionEvent& input, hknpCharacterObjectInteractionResult& output);

		void fireShapeChanged(const hknpShape* shape) const;

		void fireTriggerVolumeInteraction(hknpBodyId triggerBodyId, hknpShapeKey triggerShapeKey,
			hknpTriggerVolumeEvent::Status status) const;

		//
		// Trigger volume handling
		//

		/// Updates the list of overlapping triggers with the list of triggers seen in the last integration and fires
		/// trigger volume interaction callbacks.
		void updateOverlappingTriggers(const hkLocalArray<TriggerVolume>& triggersSeen);

		/// Updates the list of trigger volumes seen this frame with the trigger hit list returned by worldLinearCast.
		/// Only hits closer than maxFraction will be considered.
		static void updateTriggersSeen(const hkLocalArray<TriggerVolumeHit>& triggerHits, hkReal maxFraction,
			hkLocalArray<TriggerVolume>& triggersSeen);

	protected:

		/// The manifold is the representation of the immediate surroundings of the character expressed as a collection
		/// of contact points with nearby bodies. Points in the manifold are current contacts (points within
		/// m_keepDistance) or potential ones (within the threshold given by m_keepContactTolerance or intersecting the
		/// projected character trajectory).
		hkArray<hknpCollisionResult> m_manifold;

		/// See hknpCharacterProxyCinfo for details.
		const hknpShape* m_shape;

		/// See hknpCharacterProxyCinfo for details.
		hknpBodyId m_bodyId;

		/// see hknpBody for details.
		hkUint32 m_collisionFilterInfo;

		/// See hknpCharacterProxyCinfo for details.
		hknpWorld* m_world;

		/// Current transform.
		hkTransform m_transform;

		/// Cached aabb of the character's shape.
		hkAabb m_aabb;

		/// See hknpCharacterProxyCinfo for details.
		hkVector4 m_velocity;

		/// See hknpCharacterProxyCinfo for details.
		hkVector4 m_lastDisplacement;

		/// See hknpCharacterProxyCinfo for details.
		hkReal m_dynamicFriction;

		/// See hknpCharacterProxyCinfo for details.
		hkReal m_staticFriction;

		/// See hknpCharacterProxyCinfo for details.
		hkVector4 m_up;

		/// See hknpCharacterProxyCinfo for details.
		hkReal m_keepDistance;

		/// See hknpCharacterProxyCinfo for details.
		hkReal m_keepContactTolerance;

		/// See hknpCharacterProxyCinfo for details.
		hkReal m_contactAngleSensitivity;

		/// See hknpCharacterProxyCinfo for details.
		int m_userPlanes;

		/// See hknpCharacterProxyCinfo for details.
		hkReal m_maxCharacterSpeedForSolver;

		/// See hknpCharacterProxyCinfo for details.
		hkReal m_characterStrength;

		/// See hknpCharacterProxyCinfo for details.
		hkReal m_characterMass;

		/// Listeners registered for character proxy events.
		hkArray<hknpCharacterProxyListener*> m_listeners;

		/// List of currently overlapping trigger volumes.
		hkArray<TriggerVolume> m_overlappingTriggers;

		/// See hknpCharacterProxyCinfo for details.
		hkReal m_maxSlopeCosine;

		/// See hknpCharacterProxyCinfo for details.
		hkReal m_penetrationRecoverySpeed;

		/// See hknpCharacterProxyCinfo for details.
		int m_maxCastIterations;

		/// See hknpCharacterProxyCinfo for details.
		hkBool m_refreshManifoldInCheckSupport;
};

#include <Physics/Physics/Extensions/CharacterControl/Proxy/hknpCharacterProxy.inl>


#endif // HKNP_CHARACTER_PROXY_H

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
