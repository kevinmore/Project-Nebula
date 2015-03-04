/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_WORLD_SNAPSHOT_H
#define HKNP_WORLD_SNAPSHOT_H

#include <Physics/Physics/Dynamics/World/hknpWorldCinfo.h>
#include <Physics/Physics/Dynamics/Constraint/hknpConstraintCinfo.h>

extern const hkClass hknpWorldSnapshotClass;


/// A serializable utility class that can save and restore the state of a physics world.
/// This is mainly used for debugging. For asset loading use cases, hknpPhysicsSystemData is recommended instead.
class hknpWorldSnapshot : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Creates a snapshot of the given world.
		hknpWorldSnapshot( const hknpWorld& world );

		/// Serialization constructor.
		hknpWorldSnapshot( hkFinishLoadedObjectFlag flag );

		/// Destructor.
		~hknpWorldSnapshot();

		/// Save the snapshot to a file.
		void save( const char* filename ) const;

		/// Returns the stored world construction information.
		HK_FORCE_INLINE const hknpWorldCinfo& getWorldCinfo() const { return m_worldCinfo; }

		/// Create a world from the snapshot.
		hknpWorld* createWorld( const hknpWorldCinfo& worldCinfo ) const;

	protected:

		hknpWorldCinfo					m_worldCinfo;		///< Serialized world construction info structure
		hkArray< hknpBody >				m_bodies;			///< Storage for the serialized bodies
		hkArray< hkStringPtr >			m_bodyNames;		///< Storage for the serialized body names
		hkArray< hknpMotion >			m_motions;			///< Storage for the serialized motions
		hkArray< hknpConstraintCinfo >	m_constraints;		///< Storage for the serialized constraints
};

#endif // HKNP_WORLD_SNAPSHOT_H

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
