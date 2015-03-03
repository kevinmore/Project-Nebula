/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COLLISION_QUERY_COLLECTOR_H
#define HKNP_COLLISION_QUERY_COLLECTOR_H

struct hknpCollisionResult;


/// Interface for collecting hits during a collision query.
class hknpCollisionQueryCollector
{
	public:

		/// Collector hints.
		enum Hints
		{
			/// Hint informing the collision sub-system that this collector can stop collecting after the first valid hit.
			/// Useful e.g. for speeding up line of sight checks using ray cast.
			HINT_STOP_AT_FIRST_HIT = 1
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCollisionQueryCollector );

		/// Constructor.
		hknpCollisionQueryCollector()
		:	m_hints(0)
		{
			m_earlyOutHitFraction = hkSimdReal_Max;
		}

		/// Destructor.
		virtual ~hknpCollisionQueryCollector() {}

		/// Add a new hit to the collector.
		virtual void addHit( const hknpCollisionResult& hit ) = 0;

		/// Returns TRUE if the collector currently stores at least one valid hit.
		virtual bool hasHit() const = 0;

		/// The number of consecutive hits returned by getHits().
		virtual int getNumHits() const = 0;

		/// Returns a pointer to a consecutive block of hits. Use getNumHits() to get the number of hits stored in this
		/// block.
		virtual const hknpCollisionResult* getHits() const = 0;

		/// Call this method before re-using the collector for a new query.
		virtual void reset() = 0;

		/// Read-access to m_earlyOutHitFraction.
		/// This value is only of importance to users who implement their own collector and/or their own shape incl.
		/// its collision query methods.
		HK_FORCE_INLINE hkSimdReal getEarlyOutHitFraction() const
		{
			return m_earlyOutHitFraction;
		}

	public:

		/// See \a Hints enum.
		int m_hints;

	protected:

		/// The "Early Out Hit Fraction" is a performance optimization that allows to ignore hits that are further away
		/// than any previous valid hit early on in a collision query.
		/// This value is only of importance to users who implement their own collector and/or their own shape incl.
		/// its collision query methods.
		/// Some notes if you implement your own collector:
		/// - Make sure that this value is set to hkSimdReal_Max in the constructor and the implementation of the
		///   reset() method.
		/// - If your collector is only interested in the closest hit then you should update this value to
		///   hknpCollisionResult::m_fraction during addHit() for every hit that is closer than the currently closest
		///   stored hit.
		/// - If your collector is only interested in 'any' hit (and not all of them) then you should update this value
		///   to hkSimdReal_0 during addHit() to force the collision query to abort right away.
		hkSimdReal m_earlyOutHitFraction;

		friend class hknpShapeQueryInterface;
};


#endif // HKNP_COLLISION_QUERY_COLLECTOR_H

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
