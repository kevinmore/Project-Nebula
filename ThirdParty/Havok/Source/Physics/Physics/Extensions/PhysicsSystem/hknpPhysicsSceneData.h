/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_PHYSICS_SCENE_H
#define HKNP_PHYSICS_SCENE_H

#include <Physics/Physics/Extensions/PhysicsSystem/hknpPhysicsSystem.h>

extern const hkClass hknpRefWorldCinfoClass;
extern const hkClass hknpPhysicsSceneDataClass;

/// A referenced counted hknpWorldCinfo.
class hknpRefWorldCinfo : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		HK_FORCE_INLINE hknpRefWorldCinfo() {}
		HK_FORCE_INLINE hknpRefWorldCinfo( hkFinishLoadedObjectFlag flag )
			:	hkReferencedObject(flag),
			m_info(flag) {}

	public:

		hknpWorldCinfo m_info;
};

/// A serializable description of a physics scene,
/// composed of a number of physics systems and an optional world construction info.
class hknpPhysicsSceneData : public hkReferencedObject
{
	//+version(1)

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Empty constructor.
		hknpPhysicsSceneData();

		/// Serialization constructor.
		hknpPhysicsSceneData( hkFinishLoadedObjectFlag f );

		/// Destructor.
		~hknpPhysicsSceneData();

		/// Add a world and its contents to the scene.
		void addWorld( const hknpWorld* world );

		/// Get the world construction info.
		HK_FORCE_INLINE const hknpWorldCinfo *getWorldCinfo() const;

		/// Set the world construction info.
		HK_FORCE_INLINE void setWorldCinfo( const hknpWorldCinfo *info );

		/// Returns a physics system with the given name, or HK_NULL.
		hknpPhysicsSystemData* getSystemByName(const char* name);

		/// Looks for a rigid body by name.
		hknpBodyId getBodyByName(hkStringPtr name) const;

		/// Retrieves the body cInfo associated with the given ID.
		const hknpBodyCinfo* getBodyCinfo(hknpBodyId bodyId) const;
		hknpBodyCinfo* accessBodyCinfo(hknpBodyId bodyId);

		/// Attempts to remove the given shape from the referenced objects.
		/// This will only succeed if the shape is not used by any body.
		void tryRemoveShape(const hknpShape* shape);

		/// Locates the instance owning the given body ID.
		const hknpPhysicsSystemData* getBodySystem(hknpBodyId& bodyId) const;
		hknpPhysicsSystemData* accessBodySystem(hknpBodyId& bodyId);

	public:

		/// A set of physics system datas.
		hkArray< hkRefPtr<hknpPhysicsSystemData> > m_systemDatas;

	protected:

		/// An optional world construction info.
		hknpRefWorldCinfo* m_worldCinfo;
};

#include <Physics/Physics/Extensions/PhysicsSystem/hknpPhysicsSceneData.inl>


#endif // HKNP_PHYSICS_SCENE_H

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
