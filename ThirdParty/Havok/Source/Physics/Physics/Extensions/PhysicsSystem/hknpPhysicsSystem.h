/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_PHYSICS_SYSTEM_H
#define HKNP_PHYSICS_SYSTEM_H

#include <Physics/Physics/Dynamics/World/hknpWorld.h>
#include <Physics/Physics/Dynamics/Constraint/hknpConstraintCinfo.h>

extern const hkClass hknpPhysicsSystemDataClass;
class hknpPhysicsSystem;

/// A serializable construction info for a system of bodies and constraints.
/// Notes:
///   - All IDs in the construction info structures refer to local arrays, not world arrays.
///   - This keeps an extra reference to shapes in order for de-serialization to work.
///     Other objects can also be referenced in the same way if necessary.
class hknpPhysicsSystemData : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Empty constructor.
		hknpPhysicsSystemData() {}

		/// Serialization constructor.
		hknpPhysicsSystemData( hkFinishLoadedObjectFlag f );

		/// Destructor
		virtual ~hknpPhysicsSystemData();

		/// Utility method to add construction info for a specific set of bodies, and any constraints between them.
		void addBodies( const hknpWorld* world, const hknpBodyId* bodyIds, int numBodyIds );

		/// Utility method to add construction info for all bodies in a world, and any constraints between them.
		/// A mask can be provided to select only the those bodies whose m_flags & mask != 0.
		void addWorld( const hknpWorld* world, hknpBodyFlags bodyMask = hknpBody::FLAGS_MASK );

		/// Returns the index of a body for a given name or invalid value if not found.
		hknpBodyId findBodyByName( const char* name ) const;

		/// Removes the given body
		void removeBody(hknpBodyId bodyId);

		/// Removes the given motion
		void removeMotion(hknpMotionId motionId);

		/// Removes the given motion property
		void removeMotionProperty(hknpMotionPropertiesId propertyId);

		/// Removes the given material
		void removeMaterial(hknpMaterialId materialId);

		/// Remove unused motions, motions properties and materials from the physics system data.
		void collapse();

		/// Sets the collision info in the system's body construction infos to disable collision between the constrained
		/// bodies, if a hknpGroupCollisionFilter is used in the world.
		void disableCollisionsBetweenConstraintBodies( int groupFilterSystemGroup );

		/// Returns true if the given motion info is unique (i.e. used by only one rigid body)
		bool isUnique(hknpMotionId motionId) const;

		/// Adds a new motion
		hknpMotionId addMotion(const hknpMotionCinfo& newMotion);

		/// Assert that all IDs map to the local arrays and that nothing is unreferenced.
		virtual void checkConsistency() const;

	private:

		// This class is non-copyable.
		hknpPhysicsSystemData( const hknpPhysicsSystemData& );
		hknpPhysicsSystemData& operator=( const hknpPhysicsSystemData& );

	public:

		hkArray< hknpMaterial >			m_materials;
		hkArray< hknpMotionProperties >	m_motionProperties;
		hkArray< hknpMotionCinfo >		m_motionCinfos;
		hkArray< hknpBodyCinfo >		m_bodyCinfos;
		hkArray< hknpConstraintCinfo >	m_constraintCinfos;

		/// Stores a reference to shapes.
		/// (Otherwise they would not be deserialized correctly, since hknpBodyCinfo::m_shape is not reference counted).
		hkArray< hkRefPtr<const hkReferencedObject> > m_referencedObjects;

		/// Optional name for the system.
		hkStringPtr m_name;
};


/// A non-serializable runtime instance of a hknpPhysicsSystemData in a world.
class hknpPhysicsSystem : public hkReferencedObject
{
	public:

		enum Flags
		{
			/// If this flag is enabled, any powerable constraint datas (hinge, prismatic, ragdoll) will be cloned
			/// so that they can work independently if motors are applied.
			CLONE_POWERABLE_CONSTRAINT_DATAS	= 1 << 0,

			/// If this flag is enabled, the constraints will always have their applied impulses exported after solving.
			FORCE_EXPORTABLE_CONSTRAINTS		= 1 << 1,

			/// Default flags.
			DEFAULT_FLAGS = CLONE_POWERABLE_CONSTRAINT_DATAS
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Constructor.
		/// Allocates everything from the shared system data, and adds them to the world at the given transform
		/// unless DO_NOT_ADD_BODY is specified in the addition flags.
		hknpPhysicsSystem(
			const hknpPhysicsSystemData* data, hknpWorld* world, const hkTransform& transform,
			hknpWorld::AdditionFlags additionFlags = 0, Flags flags = DEFAULT_FLAGS );

		/// Destructor.
		/// Destroys all bodies, motions and constraints that were created by this system,
		/// removing them from the world if present.
		virtual ~hknpPhysicsSystem();

		/// Get the underlying shared data.
		HK_FORCE_INLINE const hknpPhysicsSystemData* getData() const;

		/// Get the system's world.
		HK_FORCE_INLINE const hknpWorld* getWorld() const;
		HK_FORCE_INLINE hknpWorld* accessWorld();

		/// Get the body ID array. Some IDs can be invalid if the corresponding bodies have been destroyed after
		/// the system's creation.
		HK_FORCE_INLINE const hkArray< hknpBodyId >& getBodyIds() const;
		HK_FORCE_INLINE hkArray< hknpBodyId >& accessBodyIds();

		/// Get the constraints array.
		HK_FORCE_INLINE const hkArray< hknpConstraint* >& getConstraints() const;
		HK_FORCE_INLINE hkArray< hknpConstraint* >& accessConstraints();

		/// Add any bodies and constraints that are not already added to the world.
		void addToWorld( hknpWorld::AdditionFlags bodyAdditionFlags = 0,
			hknpActivationMode::Enum constraintActivationMode = hknpActivationMode::ACTIVATE );

		/// Remove all bodies and constraints from the world.
		void removeFromWorld();

		/// Utility to setup body collision filters that will disable collision between constrained bodies.
		/// Use this function if you want to disable collisions between constrained bodies on specific physics system
		/// instances. Otherwise you can use the physics system data equivalent.
		void disableCollisionsBetweenConstraintBodies( int groupFilterSystemGroup );

		// Signal handler
		void onBodyDestroyedSignal( hknpWorld* world, hknpBodyId bodyId );

	protected:

		/// Shared data.
		hkRefPtr< const hknpPhysicsSystemData >	m_data;

		/// The world in which this system was created.
		hkRefPtr< hknpWorld > m_world;

		/// The IDs of the bodies which were constructed by this system.
		/// A one-to one correspondence is kept between this array and the shared data's hknpBodyCinfo array.
		/// If a body is destroyed, the corresponding body ID in this array is set to invalid.
		hkArray< hknpBodyId > m_bodyIds;

		/// The constraints which were constructed by this system.
		/// A one-to one correspondence is kept between this array and the shared data's hknpConstraintCinfo array.
		hkArray< hknpConstraint* > m_constraints;
};


#include <Physics/Physics/Extensions/PhysicsSystem/hknpPhysicsSystem.inl>


#endif // HKNP_PHYSICS_SYSTEM_H

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
