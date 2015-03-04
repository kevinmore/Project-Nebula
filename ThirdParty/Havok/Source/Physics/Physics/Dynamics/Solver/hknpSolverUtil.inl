/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/Algorithm/Sort/hkSort.h>

namespace
{
	struct JacGridPrioritySorter
	{
		const hknpConstraintSolverSchedulerGridInfo* m_gridInfos;

		JacGridPrioritySorter(const hknpConstraintSolverSchedulerGridInfo* gridInfos)
		:	m_gridInfos(gridInfos)
		{
		}

		HK_FORCE_INLINE bool operator()(int i, int j) const
		{
			return m_gridInfos[i].getPriority() < m_gridInfos[j].getPriority();
		}
	};
}

template< typename IndexArrayT>
HK_FORCE_INLINE void HK_CALL hknpSolverUtil::sortJacGrids(const hknpConstraintSolverSchedulerGridInfo* jacGridInfos, int jacGridCount,
															IndexArrayT& sortedResultsOut)
{
	for (hkUint8 gi = 0; gi < jacGridCount; ++gi)
	{
		sortedResultsOut[gi] = gi;
	}

	hkSort(sortedResultsOut.begin(), jacGridCount, JacGridPrioritySorter(jacGridInfos));
}

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
