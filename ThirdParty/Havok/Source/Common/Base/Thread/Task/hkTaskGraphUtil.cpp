/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Task/hkTaskGraphUtil.h>
#include <Common/Base/Thread/Task/hkTaskGraph.h>
#include <Common/Base/System/Io/OStream/hkOStream.h>

// See http://www.graphviz.org/content/dot-language for details on the output format.
void HK_CALL hkTaskGraphUtil::print(const hkDefaultTaskGraph* taskGraph, hkTaskGraphUtil::TaskPrinter* taskPrinter, hkOstream& outStream)
{
	outStream.printf("digraph {\n");
	const hkArray<hkDefaultTaskGraph::TaskInfo>& taskInfos = taskGraph->m_taskInfos;
	for (int i = 0; i < taskInfos.getSize(); ++i)
	{
		const hkDefaultTaskGraph::TaskInfo& parentInfo = taskInfos[i];
		hkStringBuf parentName;
		hkStringBuf nodeAttributes;
		taskPrinter->print(parentInfo.m_task, parentName, nodeAttributes);
		outStream.printf("\t\"%s\" %s\n", parentName.cString(), nodeAttributes.cString());
		int childIndex = parentInfo.m_firstChildIndex;
		for (int j = 0; j < parentInfo.m_numChildren; ++j)
		{
			const hkDefaultTaskGraph::TaskInfo& childInfo = taskInfos[taskGraph->m_children[childIndex++].value()];
			hkStringBuf childName;
			taskPrinter->print(childInfo.m_task, childName, nodeAttributes);
			outStream.printf("\t\"%s\" -> \"%s\"\n", parentName.cString(), childName.cString());
		}
	}
	outStream.printf("}\n");
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
