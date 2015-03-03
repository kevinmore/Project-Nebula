/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_HYBRID_AABB_TREE_H
#define HKNP_HYBRID_AABB_TREE_H

#include <Common/Base/Algorithm/Collide/1AxisSweep16/hk1AxisSweep16.h>
#include <Common/Base/Types/Geometry/IntSpaceUtil/hkIntSpaceUtil.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Geometry/Internal/Algorithms/Intersect/hkcdIntersectRayAabb.h>

#define HKNP_ON_HYBRID_TREE_TIMER_ENABLED(X)

struct hkcdRay;

typedef hkUint32 hknpHTreeNodeId;

//	note: should avoid to store pointer to nodes which can change on nodeManager->reserve


/// A tree used by hknpHybridBroadPhase
template<typename KeyType, typename hkIdxType>
class hknpHybridAabbTree
{
	public:

		static const hkIdxType s_invalidIdx = hkIdxType(~0) >> 1;

		struct Node
		{
			HK_DECLARE_PLACEMENT_ALLOCATOR();

			HK_ALIGN16(hkAabb16 m_volume);

			struct NodeDataType
			{
				hkIdxType m_childIdx[2];
			};

			struct NodeLeafType
			{
				hkIdxType m_leafArrayIdx;			///< Index into the leaf data array
				hkIdxType m_leafFlg;				///< see isLeaf()
			};

			union
			{
				NodeDataType m_nodeData;
				NodeLeafType m_leafData;
			};
			int m_totalNumberOfLeafs;

			HK_FORCE_INLINE bool isLeaf() const { return m_leafData.m_leafFlg == s_invalidIdx;}
			HK_FORCE_INLINE void setLeafData(hkIdxType idx) { m_leafData.m_leafFlg = s_invalidIdx; m_leafData.m_leafArrayIdx = idx; }
			HK_FORCE_INLINE void setChild(int i, hkIdxType childIdx) { m_nodeData.m_childIdx[i] = childIdx; }
			HK_FORCE_INLINE hkIdxType getChildIdx(int i) const {return m_nodeData.m_childIdx[i];}
		};

		HK_FORCE_INLINE Node* getChild(const Node* node, int i) const { return const_cast<Node*>(m_nodeManager.getNode( node->m_nodeData.m_childIdx[i] )); }
		HK_FORCE_INLINE Node* getNode(hkIdxType nodeIdx) const { return const_cast<Node*>(m_nodeManager.getNode( nodeIdx )); }
		HK_FORCE_INLINE Node* getRootNode() const { return (m_rootNodeIdx == s_invalidIdx)? HK_NULL: const_cast<Node*>(m_nodeManager.getNode( m_rootNodeIdx ) );}
		HK_FORCE_INLINE Node* getRootNodeNonZero() const { return const_cast<Node*>(m_nodeManager.getNode( m_rootNodeIdx ) );}

		enum
		{
			NUM_SWEEP_PADDING = 2,								// see hk1AxisSweep16<T>::collidePadding2
			MAX_NUM_NODES_IN_LEAF = 32-NUM_SWEEP_PADDING-1,		// try to set it memory consumption to be a nice multiple of 128 (XBOX etc) (-1 because of sizeof(LeafData) )
		};

		struct LeafData
		{
			HK_FORCE_INLINE	void pushBack(const hkAabb16& aabb){m_aabbs[ m_numAabbs++ ] = aabb;}

			HK_FORCE_INLINE	hkAabb16& expandOne(){return m_aabbs[ m_numAabbs++ ];}

			HK_FORCE_INLINE	void clear() { m_numAabbs = 0; m_age = 0; }

			HK_FORCE_INLINE	void invalidate(int i){ m_aabbs[i].m_min[0] = hkUint16(-1); m_numAabbs--;}

			HK_FORCE_INLINE void removeAtAndCopy( int i );

			/// set 4 elements at the end of the list to invalid, needed for the radix sort
			HK_FORCE_INLINE	void addFooter();

			/// Combine all body aabbs within this leaf
			HK_FORCE_INLINE	void calcLeafAabb(hkAabb16& aabbOut) const;

			HK_FORCE_INLINE	void quickSortAabbs();

			HK_FORCE_INLINE	void incrementAge(){ m_age++; }

			int m_numAabbs;
			hkUint32 m_age;		///< criteria to select which leaf to open

			HK_ALIGN16(hkAabb16 m_aabbs[ MAX_NUM_NODES_IN_LEAF + NUM_SWEEP_PADDING ]);
		};

		// A pair of pointers
		struct PtrPair
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS, PtrPair);
			PtrPair() {}
			PtrPair(void* a, void* b) { m_a = a; m_b = b; }

			void* m_a;
			void* m_b;
		};

		// A pair of keys
		struct KeyPair
		{
			KeyPair() {}
			KeyPair(KeyType a, KeyType b) : m_a(a), m_b(b) {}
			bool operator == (const KeyPair& in) const { return (m_a == in.m_a && m_b == in.m_b); }

			KeyType m_a;
			KeyType m_b;
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpHybridAabbTree );

		/// Set num of objects per leaf. But memory is allocated for MAX_NUM_NODES_IN_LEAF anyway
#ifdef HK_PLATFORM_SPU
		hknpHybridAabbTree( int numObjPerLeaf = MAX_NUM_NODES_IN_LEAF ) {}
#else
		hknpHybridAabbTree( int numObjPerLeaf = MAX_NUM_NODES_IN_LEAF );
		~hknpHybridAabbTree();
#endif


		/// Build a tree from scratch, calls clear() initially.
		void build(hkAabb16* volumes, KeyType* userDatas, int numObjects);

		/// Remove all
		void clear();

		/// rebuild the aabb of this subtree, return the number of leaf nodes
		int fixupVolume(Node* node);

		/// return a free index into the leafData array
		HK_FORCE_INLINE	hkIdxType allocateLeafDataIdx();

		/// frees up a used data array entry
		HK_FORCE_INLINE	void deallocateLeafDataIdx(hkIdxType idx);

		int getNumObjPerLeaf() const { return m_numObjPerLeaf; }

		//
		//	QUERY FUNCTIONS

		/// collision between input aabbs.
		/// Notes:
		///   - Normally root == tree
		///   - userData would be body index in hknpWorld


		//
		//	query functions up to leaves(don't open leaves)
		//
		HK_FORCE_INLINE static void HK_CALL queryAabb(
			const hknpHybridAabbTree<KeyType, hkIdxType>* tree, hkIdxType rootNodeIdx,
			const hkAabb16& queryVolume, hkArray<hkIdxType>& leafIdxOut );

		template<class VISIT>
		HK_FORCE_INLINE static void HK_CALL castRay(
			const hknpHybridAabbTree<KeyType, hkIdxType>* tree, hkIdxType rootNodeIdx,
			const hkcdRay& rayInTreeSpace, VISIT& visitor );

		template<class VISIT>
		HK_FORCE_INLINE static void HK_CALL aabbCast(
			const hknpHybridAabbTree<KeyType, hkIdxType>* tree, hkIdxType rootNodeIdx,
			const hkcdRay& rayInTreeSpace, hkVector4Parameter halfExtentInTreeSpace, VISIT& visitor );

		template<class VISIT>
		HK_FORCE_INLINE static void HK_CALL closestPoint(
			const hknpHybridAabbTree<KeyType, hkIdxType>* tree, hkIdxType rootNodeIdx,
			hkVector4Parameter posInWorldSpace, hkVector4Parameter halfExtentInWorldSpace,
			const hkIntSpaceUtil& intSpaceUtil, VISIT& visitor );

		//	tree vs tree, generate leafs
		static void queryTreeCollision(
			const hknpHybridAabbTree<KeyType, hkIdxType>* treeA,
			const hknpHybridAabbTree<KeyType, hkIdxType>* treeB,
			hkArray<PtrPair>* HK_RESTRICT leafPairsOut );


		//
		//	query functions up to elements stored in leaves(open leaves)
		//

		static int HK_CALL queryAabbBatch(
			const hknpHybridAabbTree<KeyType, hkIdxType>* tree, const typename hknpHybridAabbTree<KeyType, hkIdxType>::Node* root,
			const hkAabb16* aabbs, const KeyType* userDatas, int numAabbs,
			KeyPair* hitArray, int hitCapacity );

		/// Check a single leaf with a set of volumes
		static int HK_CALL queryLeafWithVolumeBatch(
			const hknpHybridAabbTree<KeyType, hkIdxType>* tree, const typename hknpHybridAabbTree<KeyType, hkIdxType>::LeafData* leafData,
			const hkAabb16* volumeStart, const KeyType* userDataStart, const int* idxArray, int numIdx,
			KeyPair* hitArray, int hitCapacity );

		//	tree vs tree. Dump out keyPairs
		static void HK_CALL queryTreeCollision(
			const hknpHybridAabbTree<KeyType, hkIdxType>* treeA,
			const hknpHybridAabbTree<KeyType, hkIdxType>* treeB,
			hkBlockStream<hknpBodyIdPair>::Writer* pairsWriter, const hkAabb16* previousAabbs );

		//	returns numHits
		static int HK_CALL leafLeafCollision(
			const typename hknpHybridAabbTree<KeyType, hkIdxType>::LeafData& leafDataA,
			const typename hknpHybridAabbTree<KeyType, hkIdxType>::LeafData& leafDataB,
			const Node* nodeA, const Node* nodeB,
			KeyPair* hitArray, int hitCapacity);


		//
		//	OTHER FUNCTIONS

		static hkIdxType HK_CALL queryClosestLeaf(
			const Node* root, const hknpHybridAabbTree<KeyType, hkIdxType>* tree,
			hkUint16 point[3], hkUint32 initialDistance2 );

		static void HK_CALL collectNodes(
			const Node* root, const hknpHybridAabbTree<KeyType, hkIdxType>* tree,
			hkArray<const Node*>* internalsOut, hkArray<const Node*>* leavesOut );

		static void HK_CALL collectNodes(
			hkIdxType rootIdx, const hknpHybridAabbTree<KeyType, hkIdxType>* tree,
			hkArray<hkIdxType>* internalsOut, hkArray<hkIdxType>* leavesOut );

		//	better not to use this. Use next one returning idx.
		static void HK_CALL collectLeaves(
			const Node* root, const hknpHybridAabbTree<KeyType, hkIdxType>* tree,
			hkArray<const Node*>& leavesOut );

		static void HK_CALL collectLeaves(
			hkIdxType rootIdx, const hknpHybridAabbTree<KeyType, hkIdxType>* tree,
			hkArray<hkIdxType>& leafIdxOut );

		/// sort by min_x  (to be used in 1-axis sweep)
		static void HK_CALL quickSortAabbs( hkAabb16* aabbs, int size );

		//	assert
		void checkDuplicationOfStoredKeys();

		//	calling reserve can change the pointer to the node
		class NodeManager
		{
			public:
				HK_FORCE_INLINE NodeManager(){}

				/// set myIdx in node and returns the idx of this node in mem location
				hkIdxType pushBack(Node& node);

				///	these ptr to nodes are not valid after resized.
				Node* newNode(hkIdxType& newNodeIdx);

				HK_FORCE_INLINE void deleteNode(hkIdxType idx){ removeAt( idx ); }
				HK_FORCE_INLINE Node* getNode(hkIdxType idx);
				HK_FORCE_INLINE const Node* getNode(hkIdxType idx) const;

				void removeAt(hkIdxType nodeIdx);
				void reserve(hkIdxType size);
				void reserveFreeListBy(hkIdxType incSize);	///< make sure that there are at least incSize free elements

				void clear();

				int getNodeBufferSize() const { return m_nodes.getSize(); }

				enum
				{
					MEM_INCREASE_SIZE = 128
				};

			private:
				hkArray<Node> m_nodes;
				hkArray<hkIdxType> m_freeList;
		};

		//	NodeManager is managing the memory for nodes. Ptrs for nodes change when the size of array changes
		NodeManager m_nodeManager;

	private:

		//	returns number of leaves
		int buildInternal(hkIdxType rootIdx, hkAabb16& rootSpace, hkAabb16* volumes, KeyType* userDatas, int* idxArray, int numObjects);

	public:

		hkIdxType m_rootNodeIdx;
		int m_numLeaves;
		int m_numObjects;
		hkArray<LeafData> m_leafDatas;
		hkArray<hkIdxType> m_freeLeafDataIdx;
		int m_numObjPerLeaf;
};

#include <Physics/Internal/Collide/BroadPhase/Hybrid/hknpHybridAabbTree.inl>


#endif	// !HKNP_HYBRID_AABB_TREE_H

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
