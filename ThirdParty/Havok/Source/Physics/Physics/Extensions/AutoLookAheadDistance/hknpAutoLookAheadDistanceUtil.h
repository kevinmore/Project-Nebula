/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_AUTO_LOOK_AHEAD_DISTANCE_UTIL_H
#define HKNP_AUTO_LOOK_AHEAD_DISTANCE_UTIL_H

#include <Physics/Physics/hknpTypes.h>


/// A utility which can be used to avoid fast bodies accelerating other bodies through other bodies,
/// such as fast vehicles colliding with stationary bodies near static walls.
/// This works by detecting any bodies which may collide with each registered body during the next simulation step,
/// and sets the collision look ahead distance of those bodies based on the velocity of the registered body.
class hknpAutoLookAheadDistanceUtil : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Constructor
		hknpAutoLookAheadDistanceUtil( hknpWorld* world );

		/// Destructor
		virtual ~hknpAutoLookAheadDistanceUtil();

		/// Register a (fast) body.
		void registerBody( hknpBodyId id );

		/// Unregister a body.
		void unregisterBody( hknpBodyId id );

	protected:

		// Event handlers
		void onPreCollide( hknpWorld* world );
		void onBodyDestroyed( hknpWorld* world, hknpBodyId bodyId );

	public:

		/// A pointer to the world
		hknpWorld* m_world;

		/// A list of registered bodies.
		hkArray< hknpBodyId > m_registeredBodies;
};


#endif // HKNP_AUTO_LOOK_AHEAD_DISTANCE_UTIL_H

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
