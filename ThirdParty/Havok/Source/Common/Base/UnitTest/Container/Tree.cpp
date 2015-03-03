/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Container/Tree/hkTree.h>
#include <Common/Base/Container/Tree/hkSortedTree.h>

static void  tree()
{
    // Testing append and getdepth functionality
	typedef hkTree<int> Tree;
	{
		const int size = 4;
		Tree t;

		{
			Tree::Iter itr = 0;
			{
				for(int i = 0; i < size; ++i)
				{
					itr = t.appendChild(itr,i);
				}
			}

			if(itr)
			{
				HK_TEST(t.getDepth(itr)==3);
			}

			// Testing iterParent functionality
			{
				itr = t.iterParent(itr);
				HK_TEST(t.getDepth(itr)==2);
				itr = t.iterParent(itr);
				HK_TEST(t.getDepth(itr)==1);
				itr = t.iterParent(itr);
				HK_TEST(t.getDepth(itr)==0);
			}
		}

		//Testing getValue and iterChildren functionality
		{
			int i = 0;
			for(Tree::Iter itr = t.iterGetFirst(); itr!=HK_NULL;
				itr = t.iterChildren(itr))
			{
				HK_TEST(t.getValue(itr)==i);
				i++;
			}
		}

		// Testing of getNumChildren()
		{
			for(Tree::Iter itr = t.iterGetFirst(); itr!=HK_NULL;
				itr = t.iterChildren(itr))
			{
				if(t.iterChildren(itr)!= HK_NULL)
				 {
					HK_TEST(t.getNumChildren(itr) == 1);
				 }
				 else
				 {
					 HK_TEST(t.getNumChildren(itr) == 0);
				 }

			}
		}
		// Testing of clear functionality
		{
			t.clear();
			HK_TEST(t.iterGetFirst()==HK_NULL);
		}

	}

	//Testing Siblings using append.
	{
		Tree t;
		Tree::Iter itr_root = t.iterGetFirst();

		for(int i2 = 0; i2 < 4; ++i2)
		{
			t.appendChild(itr_root,i2+5);
		}

		int i = 5;
		for(Tree::Iter itr = t.iterGetFirst(); itr!=HK_NULL;
			itr = t.iterNext(itr))
		{
			HK_TEST(t.getValue(itr) == i);
			i++;
		}
	}

	// Testing iterNextPreOrder().
	{
		Tree t;
		Tree::Iter itr = t.iterGetFirst();

		int i;
		for(i = 0; i < 4; ++i)
		{
			t.appendChild(itr,i+5);
		}

		i = 0;
		itr = t.iterGetFirst();
		while(itr!=HK_NULL)
		{
			HK_TEST(t.getValue(itr) == i+5);
			itr = t.iterNextPreOrder(itr);
			i++;
		}

	}

	// Testing of remove().
	{
		Tree t;
		Tree::Iter itr = t.iterGetFirst();

		//Creating Child
		for(int i = 0; i < 7; ++i)
		{
			itr = t.appendChild(itr,i);
		}

		itr = t.iterParent(itr);
		int removed = t.getValue(itr);

		// Removing 2nd element from Last child.
		t.remove(itr);

		for(itr = t.iterGetFirst(); itr!=HK_NULL;
			itr = t.iterChildren(itr))
		{
			HK_TEST(t.getValue(itr) != removed);
		}
	}
}

static void  sorted_tree()
{	
	hkSortedTreeBase::Prng	prng;
	for(int bits=0; bits < 14; ++bits)
	{		
		const int				count = 1 << bits;
		hkSortedTree<hkUint32>	tree; tree.preAllocateNodes(count);
		hkArray<int>			nodes; nodes.reserve(count);
		HK_TRACE("Nodes: "<<count);
		
		for(int i=0; i<count; ++i)
		{
			nodes.pushBack(tree.insert(prng.nextUint32()));
		}

		HK_TEST(tree.checkIntegrity());

		int	iterationsSum=0;
		int	iterationsMax=0;
		int	closest;
	
		for(int i=0; i<nodes.getSize(); ++i)
		{
			int numIterations=0;
			int foundIndex = tree.find(tree.getValue(nodes[i]), closest, numIterations);
			HK_TEST(tree.getValue(nodes[i]) == tree.getValue(foundIndex));
			iterationsSum	+=	numIterations;
			iterationsMax	=	hkMath::max2(iterationsMax, numIterations);
		}
		HK_TRACE("Base O("<<(iterationsSum/nodes.getSize())<<":"<<iterationsMax<<")");
		
		for(int i=0; i<4096; ++i)
		{
			int		idx = prng.nextInt32() % nodes.getSize();
			int&	index = nodes[idx];
			tree.remove(index);
			index = tree.insert(prng.nextUint32());
		
			int numIterations=0;
			int foundIndex = tree.find(tree.getValue(index), closest, numIterations);

			HK_TEST(tree.getValue(index) == tree.getValue(foundIndex));
		}

		HK_TEST(tree.checkIntegrity());

		iterationsSum=0;
		iterationsMax=0;
	
		for(int i=0; i<nodes.getSize(); ++i)
		{
			int numIterations=0;
			int foundIndex = tree.find(tree.getValue(nodes[i]), closest, numIterations);
			HK_TEST(tree.getValue(nodes[i]) == tree.getValue(foundIndex));
			iterationsSum	+=	numIterations;
			iterationsMax	=	hkMath::max2(iterationsMax, numIterations);
		}
		HK_TRACE("After update O("<<(iterationsSum/nodes.getSize())<<":"<<iterationsMax<<")");		

		tree.optimize(nodes.getSize());

		HK_TEST(tree.checkIntegrity());
		
		iterationsSum=0;
		iterationsMax=0;
	
		for(int i=0; i<nodes.getSize(); ++i)
		{
			int numIterations=0;
			int foundIndex = tree.find(tree.getValue(nodes[i]), closest, numIterations);
			HK_TEST(tree.getValue(nodes[i]) == tree.getValue(foundIndex));
			iterationsSum	+=	numIterations;
			iterationsMax	=	hkMath::max2(iterationsMax, numIterations);
		}
		HK_TRACE("After optimize O("<<(iterationsSum/nodes.getSize())<<":"<<iterationsMax<<")");

		int	nodeIndex = tree.getFirst();
		while(nodeIndex)
		{
			const int i = nodes.indexOf(nodeIndex);
			HK_TEST(i != -1);
			nodes.removeAt(i);
			nodeIndex = tree.getNext(nodeIndex);
		}
		HK_TEST(nodes.getSize() == 0);
	}
}

int tree_main()
{
	tree();
	sorted_tree();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(tree_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
