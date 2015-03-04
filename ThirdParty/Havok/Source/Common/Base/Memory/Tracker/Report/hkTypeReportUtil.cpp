/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/Report/hkTypeReportUtil.h>

namespace // anonymous
{

struct TypeInfo
{
	const hkTrackerTypeTreeNode* m_type;
	hk_size_t m_size;
	hk_size_t m_numInstances;
	hk_size_t m_numBlocks;
};

} // anonymous


HK_FORCE_INLINE static hkBool32 _orderBySize(const TypeInfo& a, const TypeInfo& b)
{
	return a.m_size > b.m_size;
}

static void HK_CALL _dumpStackTrace(const char* text, void* context)
{
	hkOstream& stream = *(hkOstream*)context;
	stream << text;
}

/* static */void HK_CALL hkTypeReportUtil::report(hkTrackerScanSnapshot* scanSnapshot, hkOstream& stream)
{
	// Maps a block to its owner (HK_NULL if has no owner)
	hkPointerMap<Block*, Block*> ownerMap;
	// Maps a type to an index storing information about the type
	hkPointerMap<const RttiNode*, int> typeMap;
	hkArray<TypeInfo> typeInfos;
	hkArray<Block*> roots;

	const hkArray<Block*>& blocks = scanSnapshot->getBlocks();

	for (int i = 0; i < blocks.getSize(); i++)
	{
		// Every block with a class type is owned by itself
		Block* block = blocks[i];

		if (block->m_type && block->m_arraySize < 0)
		{
			const hkTrackerTypeTreeNode::Type type = block->m_type->m_type;
			if (type == hkTrackerTypeTreeNode::TYPE_NAMED || type == hkTrackerTypeTreeNode::TYPE_CLASS)
			{
				// Its owned by itself
				ownerMap.insert(block, block);
				roots.pushBack(block);
			}
		}
	}

	// Now lets traverse from these roots to find what blocks they reach that are not owned
	{
		hkArray<Block*> stack;

		for (int i = 0; i < roots.getSize(); i++)
		{
			Block* rootBlock = roots[i];

			stack.pushBack(rootBlock);

			while (stack.getSize() > 0)
			{
				Block* block = stack.back();
				stack.popBack();


				const int numRefs = block->m_numReferences;
				Block*const* refs = scanSnapshot->getBlockReferences(block);

				for (int j = 0; j < numRefs; j++)
				{
					Block* childBlock = refs[j];
					if (childBlock == HK_NULL)
					{
						continue;
					}

					Block* currentOwner = ownerMap.getWithDefault(childBlock, HK_NULL);
					if (currentOwner == HK_NULL)
					{
						ownerMap.insert(childBlock, rootBlock);
						stack.pushBack(childBlock);
					}
				}
			}
		}
	}

	// Now we know what blocks belong to the root... so now I can work out the sizes by type
	for (int i = 0; i < blocks.getSize(); i++)
	{
		Block* block = blocks[i];
		Block* owner = ownerMap.getWithDefault(block, HK_NULL);

		if (owner)
		{
			// Look up the type info
			int typeIndex = typeMap.getWithDefault(owner->m_type, -1);
			if (typeIndex < 0)
			{
				typeIndex = typeInfos.getSize();
				typeMap.insert(owner->m_type, typeIndex);

				TypeInfo& info = typeInfos.expandOne();
				info.m_type = owner->m_type;
				info.m_size = 0;
				info.m_numBlocks = 0;
				info.m_numInstances = 0;
			}

			TypeInfo& info = typeInfos[typeIndex];

			info.m_numBlocks++;
			info.m_size += block->m_size;
			// This is the root
			info.m_numInstances += (block == owner);
		}
	}

	// Order types by size
	hkSort(typeInfos.begin(), typeInfos.getSize(), _orderBySize);

	for (int i = 0; i < typeInfos.getSize(); i++)
	{
		const TypeInfo& info = typeInfos[i];

		stream << "Type: " << info.m_type->m_name << " Num instances: " << int(info.m_numInstances) << " Num blocks: " << int(info.m_numBlocks);
		stream << " Total Size: " << MemorySize(info.m_size) << "\n";
	}

	// Find the total amount of block usage

	{
		hk_size_t totalSize = 0;

		for (int i = 0; i < blocks.getSize(); i++)
		{
			// Every block with a class type is owned by itself
			Block* block = blocks[i];
			if (ownerMap.getWithDefault(block, HK_NULL))
			{
				totalSize += block->m_size;
			}
		}

		stream << "Total block size: " << MemorySize(totalSize) << "\n";

	}
	// Total memory used, and total allocated
	stream << "Total used: " << MemorySize(scanSnapshot->calcTotalUsed()) << "\n";

	{
		hkStackTracer tracer;

		// Go through the blocks at the end and dump any not owned by a type
		for (int i = 0; i < blocks.getSize(); i++)
		{
			Block* block = blocks[i];
			if (ownerMap.getWithDefault(block, HK_NULL) == HK_NULL)
			{
				hkTrackerTypeTreeNode::dumpType(block->m_type, stream);
				stream << "  ";

				stream << "Not owned: " << (void*)block->m_start <<" Size: " << MemorySize(block->m_size) << "\n";
				
				// Dump the call stack
				hkScanReportUtil::dumpAllocationCallStack(&tracer, scanSnapshot, (void*)(block->m_start), stream);
			}
		}
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
