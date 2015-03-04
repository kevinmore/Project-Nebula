/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/Report/hkHierarchyReportUtil.h>
#include <Common/Base/Container/String/hkStringBuf.h>

namespace // anonymous
{

/* 
struct TypeInfo
{
	const hkRttiParser::Node* m_type;
	hk_size_t m_size;
	hk_size_t m_numInstances;
	hk_size_t m_numBlocks;
}; */

struct HierarchyInfo
{
	int m_depth;								///< The depth
	const hkTrackerScanSnapshot::Block* m_block;		///< The block
	const char* m_name;							///< The name of the member that references this
};

struct PathEntry
{
	const hkTrackerTypeTreeNode* m_type;			///< The type
	int m_memberIndex;						///< The member index (-1 if not known)
};

struct NodeTree
{
	typedef hkTrackerTypeTreeNode RttiNode;
	struct Node
	{
		Node* findChild(const RttiNode* type, int memberIndex)
		{
			Node* child = m_child;
			while (child)
			{
				if (child->m_type == type && child->m_memberIndex == memberIndex)
				{
					return child;
				}
				child = child->m_next;
			}
			return HK_NULL;
		}
		void addChildNode(Node* childNode)
		{
			HK_ASSERT(0x34243a24, childNode->m_parent == HK_NULL);

			// Attach the child node
			childNode->m_parent = this;
			childNode->m_next = m_child;
			m_child = childNode;
		}
		int calcDepth() const
		{
			int depth = 0;
			for (const Node* node = this; node; node = node->m_parent) depth++;
			return depth;
		}

		const hkTrackerTypeTreeNode* m_type;
		int m_memberIndex;
		
		Node* m_parent;				///< The parent of this
		Node* m_child;				///< Child of this
		Node* m_next;				///< Sibling next along

		int m_numInstances; 
		hk_size_t m_totalSize;
		hk_size_t m_size;
	};
	Node* newNode()
	{
		Node* node = (Node*)m_nodeFreeList.alloc();
		node->m_parent = HK_NULL;
		node->m_child = HK_NULL;
		node->m_next = HK_NULL;
		node->m_memberIndex = -1;
		node->m_type = HK_NULL;
		node->m_totalSize = 0;
		node->m_size = 0;
		node->m_numInstances = 0;
		return node;
	}
	Node* insert(const PathEntry* path, int size)
	{
		Node* node = m_rootNode;

		for (int i = 0; i < size; i++)
		{
			const PathEntry& entry = path[i];

			// Lets see if the current node has it
			Node* childNode = node->findChild(entry.m_type, entry.m_memberIndex);
			if (!childNode)
			{
				childNode = newNode();

				childNode->m_type = entry.m_type;
				childNode->m_memberIndex = entry.m_memberIndex;

				// Attach the child node
				node->addChildNode(childNode);
			}
			node = childNode;
		}

		// The node is the leaf. Add the info
		return node;
	}

	Node* getRoot() const { return m_rootNode; }

	NodeTree():
		m_nodeFreeList(sizeof(Node), sizeof(void*), 2048)
	{
		m_rootNode = newNode();
	}
	
protected:

	Node* m_rootNode;
	hkFreeList m_nodeFreeList;
};


}

HK_FORCE_INLINE hkBool _orderNodes(const NodeTree::Node* a, const NodeTree::Node* b)
{
	if (a->m_type == b->m_type)
	{
		return a->m_memberIndex < b->m_memberIndex;
	}
	return a->m_type < b->m_type;
}

static void HK_CALL _writeHierarchySummaryCsv(hkTrackerScanSnapshot* scanSnapshot, NodeTree::Node* rootNode, hkScanReportUtil::ParentMap& parentMap, hkOstream& stream)
{
	typedef hkScanReportUtil::Block Block;
	typedef hkScanReportUtil::MemorySize MemorySize;

	hkTrackerLayoutCalculator* layoutCalc = scanSnapshot->getLayoutCalculator();

	hkArray<NodeTree::Node*> stack;
	stack.pushBack(rootNode);

	stream << "Size, Total, Num Instances, Class, Member\n";

	hkArray<NodeTree::Node*> path;
	// 
	{
		hkArray<NodeTree::Node*> children;
		while (stack.getSize() > 0)
		{
			NodeTree::Node* node = stack.back();
			stack.popBack();

			stream << MemorySize(node->m_size, MemorySize::FLAG_RAW) << ", " 
				   << MemorySize(node->m_totalSize, MemorySize::FLAG_RAW) << ", " 
				   << node->m_numInstances << ", ";
			
			if (node->m_type)
			{
				stream << "\"" << node->m_type->m_name << "\",";
			}
			else
			{
				stream << "(Block),";
			}

			path.clear();
			for (NodeTree::Node* cur = node; cur; cur = cur->m_parent) 
			{
				path.pushBack(cur);
			}

			for (int i = path.getSize() - 1; i >= 0; i--)
			{
				NodeTree::Node* cur = path[i];
				if (cur->m_type)
				{
					const hkTrackerTypeLayout* layout = layoutCalc->getLayout(cur->m_type);
					stream << layout->m_members[cur->m_memberIndex].m_name;
				}
				else
				{
					stream << "#";
				}

				if (i > 0)
				{
					stream << ".";
				}
			}
			stream << "\n";

			children.clear();
			// Enqueue the children
			{
				NodeTree::Node* child = node->m_child;
				for (; child; child = child->m_next)
				{
					children.pushBack(child);
				}
			}

			hkSort(children.begin(), children.getSize(), _orderNodes);

			for (int i = children.getSize() - 1; i >= 0; i--)
			{
				stack.pushBack(children[i]);
			}
		}
	}
}

static void HK_CALL _writeHierarchySummary(hkTrackerScanSnapshot* scanSnapshot, NodeTree::Node* rootNode, hkScanReportUtil::ParentMap& parentMap, hkOstream& stream)
{
	typedef hkScanReportUtil::Block Block;
	typedef hkScanReportUtil::MemorySize MemorySize;

	hkTrackerLayoutCalculator* layoutCalc = scanSnapshot->getLayoutCalculator();

	hkArray<NodeTree::Node*> stack;
	stack.pushBack(rootNode);

	// Find the biigest class name length
	int typeNameSpace = 7;
	{
		hkPointerMap<const Block*, const Block*>::Iterator iter = parentMap.getIterator();
		for (; parentMap.isValid(iter); iter = parentMap.getNext(iter))
		{
			const Block* block = parentMap.getKey(iter);
			if (block->m_type && block->m_type->isNamedType())
			{
				typeNameSpace = hkMath::max2(block->m_type->m_name.length(), typeNameSpace);
			}
		}
	}
	
	hkScanReportUtil::alignRight(stream, hkSubString("Size"), MemorySize::MAX_FULL_DIGITS);
	stream << " ";
	hkScanReportUtil::alignRight(stream, hkSubString("Total"), MemorySize::MAX_FULL_DIGITS);
	stream << "\n";

	// 
	{
		hkArray<NodeTree::Node*> children;
		hkStringBuf string;

		while (stack.getSize() > 0)
		{
			NodeTree::Node* node = stack.back();
			stack.popBack();

			stream << MemorySize(node->m_size, MemorySize::FORMAT_PAD_RAW);
			stream << " ";
			stream << MemorySize(node->m_totalSize, MemorySize::FORMAT_PAD_RAW);
			stream << " ";
			
			if (node->m_type)
			{
				hkScanReportUtil::alignRight(stream, node->m_type->m_name, typeNameSpace);
			}
			else
			{
				hkScanReportUtil::alignRight(stream, hkSubString("(Block)"), typeNameSpace);
			}

			const int depth = node->calcDepth();	

			// Output 
			hkScanReportUtil::appendSpaces(stream, depth);

			if (node->m_type)
			{
				const hkTrackerTypeLayout* layout = layoutCalc->getLayout(node->m_type);
				stream << layout->m_members[node->m_memberIndex].m_name;
			}
			else
			{
				if (depth > 1)
				{
					stream << "#";
				}
			}
			
			stream << " (" << node->m_numInstances << ") \n";

			children.clear();

			// Enqueue the children
			{
				NodeTree::Node* child = node->m_child;
				for (; child; child = child->m_next)
				{
					children.pushBack(child);
				}
			}

			hkSort(children.begin(), children.getSize(), _orderNodes);

			for (int i = children.getSize() - 1; i >= 0; i--)
			{
				stack.pushBack(children[i]);
			}
		}
	}
}

void HK_CALL hkHierarchyReportUtil::report(hkTrackerScanSnapshot* scanSnapshot, hkBool asCsv, const Block* rootBlock, FollowFilter* filter, hkOstream& stream)
{
	hkTrackerLayoutCalculator* layoutCalc = scanSnapshot->getLayoutCalculator();

	ParentMap parentMap;
	hkScanReportUtil::calcParentMap(scanSnapshot,rootBlock, filter, parentMap);

	NodeTree nodeTree;

	{
		ParentMap::Iterator iter = parentMap.getIterator();
		hkArray<PathEntry> path;

		for (; parentMap.isValid(iter); iter = parentMap.getNext(iter))
		{
			const Block* block = parentMap.getKey(iter);
			const Block* cur = block;

			path.clear();
			while (cur)
			{
				const Block* parentBlock = parentMap.getWithDefault(cur, HK_NULL);
				if (!parentBlock)
				{
					break;
				}

				int memberIndex = -1;
				const RttiNode* type = HK_NULL;
				
				if (parentBlock && parentBlock->m_type && parentBlock->m_arraySize < 0)
				{
					// Okay I need to see if we have a layout
					const hkTrackerTypeLayout* layout = layoutCalc->getLayout(parentBlock->m_type);

					if (layout && layout->m_fullScan == false)
					{
						type = parentBlock->m_type;
						memberIndex = scanSnapshot->findReferenceIndex(parentBlock, cur);
						HK_ASSERT(0x2a423423, memberIndex >= 0);
					}
				}

				PathEntry& entry = path.expandOne();
				entry.m_type = type;
				entry.m_memberIndex = memberIndex;
				
				// Goto the parent
				cur = parentBlock;
			}

			// Reverse the path
			if (path.getSize() > 1)
			{
				PathEntry* end = &path.back();
				for (int i = 0; i < path.getSize() / 2; i++)
				{
					hkAlgorithm::swap(path[i], end[-i]);
				}
			}

			// Insert the path
			NodeTree::Node* node = nodeTree.insert(path.begin(), path.getSize()); 

			// I can associate sums with this
			node->m_numInstances++;
			node->m_size += block->m_size;

			// Add the size to this and all of the parents
			while (node)
			{
				node->m_totalSize += block->m_size;	
				node = node->m_parent;
			}
		}
	}

	// Now to report
	if (asCsv)
	{
		_writeHierarchySummaryCsv(scanSnapshot, nodeTree.getRoot(), parentMap, stream);
	}
	else
	{
		_writeHierarchySummary(scanSnapshot, nodeTree.getRoot(), parentMap, stream);
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
