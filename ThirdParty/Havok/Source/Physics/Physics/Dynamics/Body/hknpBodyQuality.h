/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_BODY_QUALITY_H
#define HKNP_BODY_QUALITY_H

#include <Physics/Physics/hknpTypes.h>


/// A quality structure for bodies.
/// This allows for control over various features of the engine on a per-body basis.
class hknpBodyQuality
{
	public:

		/// Flags which request special behaviors for a pair of bodies. When two bodies collide, the requested flags
		/// (m_requestedFlags) of the body with the quality with the highest priority (m_priority) will be combined with
		/// the supported flags (m_supportedFlags) of the other. See combineBodyQualities().
		enum FlagsEnum
		{
			/// Disable caching of manifold information between collision detection steps.
			/// This reduces memory requirements, but typically results in increased CPU cost and worse friction
			/// (objects can slowly slide down slopes).
			DROP_MANIFOLD_CACHE = 1<<0,

			/// Disable caching of contact normals if the other object collides with a vertex or an edge, instead of a face.
			/// If you don't set this flag, you might get strange normals if an object slides over a mesh.
			/// - This is automatically set in collision caches if either simple or motion welding is enabled
			FORCE_GSK_EXECUTION = 1<<1,

			/// Manifolds will consist of only a single closest point between the two shapes.
			/// This is faster to compute than a full manifold, but does not create enough contacts to support most shapes.
			/// - This is not compatible with the SILHOUETTE_COLLISIONS flag.
			FORCE_GSK_SINGLE_POINT_MANIFOLD = 1<<2,

			/// Collide with infinite planes if the other body's shape provides them (for concave triangles).
			/// This gives faster collision detection with no edge collisions, but significantly increases the
			/// likelihood of ghost impulses. Will be ignored on SPU.
			ALLOW_CONCAVE_TRIANGLE_COLLISIONS = 1<<3,

			/// Perform extra collision detection between the silhouette of the first object and the opposing face of the
			/// second object. This greatly reduces tunneling artifacts for long thin and highly tessellated objects.
			ENABLE_SILHOUETTE_MANIFOLDS = 1<<4,

			/// Rebuild contact Jacobians after every solver sub step, giving more accurate contact constraints but with
			/// increased CPU cost. This reduces the likelihood of ghost impulses by a factor of the number of sub steps.
			ENABLE_LIVE_JACOBIANS = 1<<5,

			/// Perform extra iterations at the end of each contact solving sub step. This can dramatically remove
			/// tunneling artifacts for long thin bodies. It should not make a big difference for relatively round bodies.
			USE_HIGHER_QUALITY_CONTACT_SOLVING = 1<<6,

			/// Use neighbor welding (if available). See hknpNeighborWeldingModifier.
			ENABLE_NEIGHBOR_WELDING = 1<<7,

			/// Use motion welding. See hknpMotionWeldingModifier.
			ENABLE_MOTION_WELDING = 1<<8,

			/// Use triangle welding. See hknpTriangleWeldingModifier.
			ENABLE_TRIANGLE_WELDING = 1<<9,

			/// Helper mask used to check if a body requests any kind of welding.
			ANY_WELDING = ENABLE_NEIGHBOR_WELDING | ENABLE_MOTION_WELDING | ENABLE_TRIANGLE_WELDING,

			/// The collision cache will only hold 16 bits, so the next flags will not end up in the collision cache.
			FIRST_NON_CACHABLE_FLAG = 1<<16,

			/// Clip angular velocity to stop long thin objects from tunneling through landscapes.
			/// The maximum angle a body can rotate during each step is calculated based on the minimum angle
			/// between any two neighboring faces of its shape.
			CLIP_ANGULAR_VELOCITY = 1<<16,

			/// Expand body AABBs by considering only the current and predicted transforms, rather than sweeping the
			/// body along its motion. This gives tighter expanded AABBs and thus fewer manifolds for spinning bodies,
			/// but could allow (for example) a spinning blade to rotate through wall next to it.
			USE_DISCRETE_AABB_EXPANSION = 1<<17,

			/// If set, the friction for a convex body colliding with multiple child pieces (e.g. triangles) of a
			/// composite shape (e.g. mesh) will be simplified to a single contact friction.
			MERGE_FRICTION_JACOBIANS = 1<<18,
		};

		typedef hkFlags<FlagsEnum, hkUint32> Flags;

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpBodyQuality );
		HK_DECLARE_REFLECTION();
		HK_DECLARE_POD_TYPE();

		/// Empty constructor.
		hknpBodyQuality();

		/// Initialize the quality with default settings.
		/// All flags are supported. No flags are requested.
		/// unitScale should be the units of a 1 meter cube (see hknpWorldCinfo::m_unitScale).
		void initialize( hkReal unitScale = 1.0f );

		/// Comparison operator.
		HK_FORCE_INLINE bool operator==( const hknpBodyQuality& other ) const;

		/// Combine two body qualities.
		/// Return the one with higher priority, and the set of requested flags that are supported.
		static HK_FORCE_INLINE const hknpBodyQuality* combineBodyQualities(
			const hknpBodyQuality* qA, const hknpBodyQuality* qB, Flags* flagsOut );

	public:

		/// Decides which object of the two collision partners becomes the higher priority object.
		/// E.g. a car is more important than debris, a car is more important than the landscape but a grenade is more
		/// important than the car. The basic idea is which object defines the quality of the interaction.
		int m_priority;

		/// Flags controlling which features are supported by this body (if this is the lower priority body)
		Flags m_supportedFlags;

		/// Flags controlling which features are requested by this body.
		Flags m_requestedFlags;

		/// When LIVE_JACOBIANS is enabled, this specifies the maximum distance that a body can move without causing
		/// the Jacobian to be rebuilt. Defaults to 0.1.
		hkReal m_liveJacobianDistanceThreshold;

		/// When LIVE_JACOBIANS is enabled, this specifies the maximum angle (in radians) that a body can rotate without
		/// the Jacobian to be rebuilt. Defaults to 0.5.
		hkReal m_liveJacobianAngleThreshold;
};

#include <Physics/Physics/Dynamics/Body/hknpBodyQuality.inl>


#endif // !HKNP_BODY_QUALITY_H

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
