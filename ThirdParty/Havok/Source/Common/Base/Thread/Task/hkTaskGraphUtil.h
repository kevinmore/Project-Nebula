/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_TASK_GRAPH_UTILS_H
#define HK_TASK_GRAPH_UTILS_H

struct hkDefaultTaskGraph;
class hkTask;
class hkStringBuf;

class hkTaskGraphUtil
{
	public:

		/// You must provide an object implementing this interface to the print function to generate the names for your
		/// tasks.
		class TaskPrinter
		{
			public:

				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE_CLASS, TaskPrinter);

				virtual void print(const hkTask* job, hkStringBuf& nodeNameOut, hkStringBuf& nodeAttributesOut) = 0;

				virtual ~TaskPrinter() {}
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE_CLASS, hkTaskGraphUtil);

		/// Prints the job group as a directed graph in .gv format.
		/// The output can be turned into an image using Graphviz's dot.exe.
		static void HK_CALL print(const hkDefaultTaskGraph* taskGraph, TaskPrinter* taskPrinter, hkOstream& outStream);
};

#endif // HK_TASK_GRAPH_UTILS_H

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
