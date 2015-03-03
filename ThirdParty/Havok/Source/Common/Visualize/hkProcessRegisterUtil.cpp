/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>
#include <Common/Visualize/hkProcessRegisterUtil.h>
#include <Common/Visualize/Process/hkDebugDisplayProcess.h>
#include <Common/Visualize/Process/hkInspectProcess.h>
#include <Common/Visualize/Process/hkMemorySnapshotProcess.h>
#include <Common/Visualize/Process/hkRemoteObjectProcess.h>
#include <Common/Visualize/Process/hkStatisticsProcess.h>

void HK_CALL hkProcessRegisterUtil::registerAllCommonProcesses()
{
	hkDebugDisplayProcess::registerProcess();
	hkStatisticsProcess::registerProcess();
	hkInspectProcess::registerProcess();
	hkRemoteObjectProcess::registerProcess();
	hkMemorySnapshotProcess::registerProcess();
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
