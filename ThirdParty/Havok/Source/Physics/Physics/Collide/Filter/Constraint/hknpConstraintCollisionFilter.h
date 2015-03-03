/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONSTRAINT_COLLISION_FILTER_H
#define HKNP_CONSTRAINT_COLLISION_FILTER_H

#include <Physics/Physics/Collide/Filter/Pair/hknpPairCollisionFilter.h>

extern const hkClass hknpConstraintCollisionFilterClass;

/// This filter allows to disable collisions between two bodies if they are connected through a constraint
/// (other than a contact constraint).
///
/// You can supply a child filter which will be queried first. A collision will be reported only if both two filters
/// (child filter and this constraint filter) return true.
///
/// The filter is able to subscribe to constraint signals and thus automatically add and remove entity pairs upon
/// addition and removal of constraints.
/// The filter provides a utility function updateFromWorld() which allows to sync itself with the world's current state
/// in regards to constraints.
class hknpConstraintCollisionFilter : public hknpPairCollisionFilter
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Constructor. The \a childFilter (if set to != HK_NULL) will be executed prior to this constraint filter.
		hknpConstraintCollisionFilter( const hknpCollisionFilter* childFilter = HK_NULL );

		/// Serialization constructor.
		hknpConstraintCollisionFilter( hkFinishLoadedObjectFlag flag );

		/// Destructor.
		~hknpConstraintCollisionFilter();

		/// Sync the filter with the supplied world's current state, i.e. add all current constraints to its internal
		/// table of disabled body pairs.
		void updateFromWorld( hknpWorld* world );

		/// Subscribe this filter to a world's constraint signals.
		void subscribeToWorld( hknpWorld* world );

		/// Unsubscribe this filter from the world it is subscribed to, if any.
		void unsubscribeFromWorld();

		// Signal handlers.
		void onConstraintAddedSignal( hknpWorld* world, hknpConstraint* constraint );
		void onConstraintRemovedSignal( hknpWorld* world, hknpConstraint* constraint );

	protected:

		hknpWorld* m_subscribedWorld;	//+nosave
};


#endif // HKNP_CONSTRAINT_COLLISION_FILTER_H

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
