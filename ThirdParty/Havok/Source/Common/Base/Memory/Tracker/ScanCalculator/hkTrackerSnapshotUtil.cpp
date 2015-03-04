/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/ScanCalculator/hkTrackerSnapshotUtil.h>
#include <Common/Base/Memory/Tracker/ScanCalculator/hkTrackerScanCalculator.h>

static hkTrackerScanSnapshot* makeSnapshot(hkTrackerSnapshot& snapshot)
{
	snapshot.init();
#if defined(HK_PLATFORM_PS3_PPU)
	HK_ASSERT(0x2423432, snapshot.checkConsistent() == HK_SUCCESS);
#endif
	return hkTrackerSnapshotUtil::createSnapshot(snapshot);
}

hkTrackerScanSnapshot* hkTrackerSnapshotUtil::createSnapshot()
{
	hkTrackerSnapshot snapshot; // use malloc
	return makeSnapshot(snapshot);
}

hkTrackerScanSnapshot* hkTrackerSnapshotUtil::createSnapshot(hkMemoryAllocator* mem)
{
	hkTrackerSnapshot snapshot(mem);
	return makeSnapshot(snapshot);
}

hkTrackerScanSnapshot* hkTrackerSnapshotUtil::createSnapshot(const hkTrackerSnapshot& snapshot, hkTrackerLayoutCalculator* layoutCalc)
{
	hkTrackerScanCalculator scanCalc;
	if (layoutCalc == HK_NULL)
	{
		hkTrackerTypeTreeCache* typeCache = new hkTrackerTypeTreeCache;
		layoutCalc = new hkTrackerLayoutCalculator(typeCache);
		typeCache->removeReference();

		hkTrackerScanSnapshot* scanSnapshot = scanCalc.createSnapshot(&snapshot, layoutCalc);
		layoutCalc->removeReference();

		return scanSnapshot;
	}
	else
	{
		return scanCalc.createSnapshot(&snapshot, layoutCalc);
	}
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
