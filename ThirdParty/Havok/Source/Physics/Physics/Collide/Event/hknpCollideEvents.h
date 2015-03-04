/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_COLLIDE_EVENTS_H
#define HKNP_COLLIDE_EVENTS_H

#include <Physics/Physics/Dynamics/World/Events/hknpEvents.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterial.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>

#include <Common/Base/Container/CommandStream/hkCommandStream.h>
#include <Geometry/Internal/Types/hkcdManifold4.h>

class hkOstream;


/// Event raised every time a manifold between two bodies is created or destroyed.
struct hknpManifoldStatusEvent : public hknpBinaryBodyEvent
{
	public:

		enum Status
		{
			MANIFOLD_CREATED,		///< A new manifold has been created.
			MANIFOLD_DESTROYED,		///< An existing manifold has been destroyed.
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpManifoldStatusEvent );

		/// Constructor
		HK_FORCE_INLINE hknpManifoldStatusEvent(
			hknpBodyId idA, hknpShapeKey keyA,
			hknpBodyId idB, hknpShapeKey keyB,
			hknpManifoldCollisionCache* manifoldCache, Status status );

		/// Check if the manifold involves a trigger volume material.
		HK_FORCE_INLINE hkBool32 involvesTriggerVolume() const;

		/// Returns the status as a user readable string.
		const char* getStatusAsString() const;

		/// Print this.
		void printCommand( hknpWorld* world, hkOstream& stream ) const;

	public:

		hknpShapeKey m_shapeKeys[2];		///< The closest/contacting shape key of each body.
		hkEnum<Status, hkUint8> m_status;	///< The manifold status. See Status for details.
		hknpManifoldCollisionCache* m_manifoldCache;	///< Pointer to the cache. Not available on SPU.
};

HKNP_DECLARE_EVENT_DISCRIMINATOR( hknpManifoldStatusEvent, hknpEventType::MANIFOLD_STATUS );


/// Event raised every time a manifold between 2 bodies is updated.
/// Note: you can access the Jacobians using
/// m_manifoldCache->m_manifoldSolverInfo.m_contactJacobian and m_manifoldIndex.
struct hknpManifoldProcessedEvent : public hknpBinaryBodyEvent
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpManifoldProcessedEvent );

		/// Constructor
		HK_FORCE_INLINE hknpManifoldProcessedEvent(
			hknpBodyId idA, hknpShapeKey shapeKeyA,
			hknpBodyId idB, hknpShapeKey shapeKeyB );

		/// Check if the manifold involves a trigger volume material.
		HK_FORCE_INLINE hkBool32 involvesTriggerVolume() const;

		/// Flip body A and B. This is only a user call and not called by the engine.
		HK_FORCE_INLINE void flip();

		/// Print this.
		void printCommand( hknpWorld* world, hkOstream& stream ) const;

	public:

		hknpShapeKey m_shapeKeys[2];	///< The closest/contacting shape key of each body.
		hkUint8 m_numContactPoints;		///< The number of contact points in the manifold.
		hkUint8 m_isNewManifold;		///< Set to true if this is the first process event for this manifold.
		hkUint8 m_flipped;				///< Internal flag, set to 0 by default, tracks number of user calls to flip.
		hkcdManifold4 m_manifold;		///< A copy of the manifold data.
		hknpManifoldCollisionCache* m_manifoldCache;	///< Pointer to the cache. Not available on SPU.
};

HKNP_DECLARE_EVENT_DISCRIMINATOR( hknpManifoldProcessedEvent, hknpEventType::MANIFOLD_PROCESSED );


/// Event raised every time a body enters or leaves a trigger volume, or if the trigger volume or a body inside changes.
struct hknpTriggerVolumeEvent : public hknpBinaryBodyEvent
{
	public:

		/// The status of the interaction between the trigger volume and the other body.
		enum Status
		{
			STATUS_ENTERED,	///< A body has entered the trigger volume.
			STATUS_EXITED,	///< A body has exited the trigger volume.
			STATUS_UPDATED	///< The trigger volume or a body inside the trigger volume has changed (e.g. material modified, shape changed, etc.)
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, hknpTriggerVolumeEvent );

		/// Constructor
		hknpTriggerVolumeEvent(
			hknpBodyId idA, hknpShapeKey keyA,
			hknpBodyId idB, hknpShapeKey keyB,
			Status status )
		:	hknpBinaryBodyEvent( hknpEventType::TRIGGER_VOLUME, sizeof(*this), idA, idB )
		,	m_status(status)
		{
			m_shapeKeys[0] = keyA;
			m_shapeKeys[1] = keyB;
		}

		/// Print this.
		void printCommand( hknpWorld* world, hkOstream& stream ) const;

	public:

		hkEnum<Status, hkUint8> m_status;	///< The event status. See Status for details.
		hknpShapeKey m_shapeKeys[2];		///< The penetrating/closest shape key of each body.
};

HKNP_DECLARE_EVENT_DISCRIMINATOR( hknpTriggerVolumeEvent, hknpEventType::TRIGGER_VOLUME );


#include <Physics/Physics/Collide/Event/hknpCollideEvents.inl>

#endif // !HKNP_COLLIDE_EVENTS_H

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
