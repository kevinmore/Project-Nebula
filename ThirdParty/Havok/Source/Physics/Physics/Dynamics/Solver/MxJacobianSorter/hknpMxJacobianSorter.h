/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_JACOBIAN_CONFIGULATOR_H
#define HKNP_JACOBIAN_CONFIGULATOR_H

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactSolver.h>
#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobian.h>

#if defined(HK_PLATFORM_SPU)
#	define HK_JAC_SORTER_LIMIT_ENTRY_AGE 16	// if set, limits the age of a jacobian
#endif


class hknpMxJacobianSorter
{
	public:
		//+hk.MemoryTracker(ignore=True)

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpMxJacobianSorter );

		HK_FORCE_INLINE	hknpMxJacobianSorter( hknpConstraintSolverJacobianWriter* HK_RESTRICT jacWriter );

		HK_FORCE_INLINE	~hknpMxJacobianSorter();

		static HK_FORCE_INLINE hkUint32 HK_CALL calcBodyIdsHashCode( const hknpBody& bodyA, const hknpBody& bodyB );
		static HK_FORCE_INLINE hkUint32 HK_CALL calcMotionIdsHashCode(hknpMotionId motionIdA, hknpMotionId motionIdB);

		///
		HK_FORCE_INLINE	int _getJacobianLocation( hkUint32 bodyIdsHashCode, HK_PAD_ON_SPU(hknpMxContactJacobian*)* jacOut, HK_PAD_ON_SPU(hknpMxContactJacobian*)* jacOnPpuOut);

		int getJacobianLocation( hkUint32 bodyIdsHashCode, HK_PAD_ON_SPU(hknpMxContactJacobian*)* jacOut, HK_PAD_ON_SPU(hknpMxContactJacobian*)* jacOnPpuOut);

		/// Call this just before getJacobianLocation(), if you are using the same bodies as last time
		HK_FORCE_INLINE void hintUsingSameBodies();

		/// Call this to undo an unused hintUsingSameBodies()
		HK_FORCE_INLINE void resetHintUsingSameBodies();

		/// Flush data, call this if you intent to reuse an instance of this class, otherwise not necessary,
		/// however in both cases you still have to flush the underlying writer
		HK_FORCE_INLINE	void flush();

		HK_ON_DEBUG( void printStats(); )

	protected:

		HK_FORCE_INLINE void flushElementRange( int startIndex, int numElementsToFlush );

		struct MxJacobianElement
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpMxJacobianSorter::MxJacobianElement );

			HK_FORCE_INLINE		void flush( hknpConstraintSolverJacobianWriter* jacWriter);

			HK_FORCE_INLINE		int initAndReserveEntry(hkUint32 bodyIdsHashCode, int time, hknpConstraintSolverJacobianWriter* HK_RESTRICT jacWriter, HK_PAD_ON_SPU(hknpMxContactJacobian*)* jacOut, HK_PAD_ON_SPU( hknpMxContactJacobian*)* HK_RESTRICT jacOnPpuOut);

			//	return mxJacIdx or -1 on failure
			HK_FORCE_INLINE		int checkAndReserveEntry(hkUint32 bodyIdsHashCode, HK_PAD_ON_SPU(hknpMxContactJacobian*)* jacOut, HK_PAD_ON_SPU(hknpMxContactJacobian*)* HK_RESTRICT  jacOnPpuOut );

			HK_ALIGN16(int m_numJacUsed);			// counts the number of jacobians, once it reaches hk4xVector4::mxLength, the element is flushed
			hkUint32 m_usedBodiesHashCode;
#if defined(HK_JAC_SORTER_LIMIT_ENTRY_AGE)
			int m_birthDay;
#endif

			hknpMxContactJacobian* m_jacs;
			HK_ON_SPU(hknpMxContactJacobian* m_jacsPpu;)
		};

		enum
		{
			MAX_NUM_OPEN_MX_JAC = HKNP_MAX_NUM_MANIFOLDS_PER_BATCH,	// must be a power of 2
		};

		MxJacobianElement m_mxJacElement[ MAX_NUM_OPEN_MX_JAC ];

		// offsets the search so that we avoid very likely collisions in case of convex-mesh collisions
		HK_PAD_ON_SPU(int) m_searchOffset;

		// the number of open mx jacobians
		HK_PAD_ON_SPU(int) m_numOpenElements;

		// the index of our last hit
		HK_PAD_ON_SPU(int) m_lastSearchHit;

		HK_PAD_ON_SPU(hknpConstraintSolverJacobianWriter*) m_jacWriter;

#if defined(HK_JAC_SORTER_LIMIT_ENTRY_AGE)
		HK_PAD_ON_SPU(int) m_time;
#endif
};

#include <Physics/Physics/Dynamics/Solver/MxJacobianSorter/hknpMxJacobianSorter.inl>

#endif

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
