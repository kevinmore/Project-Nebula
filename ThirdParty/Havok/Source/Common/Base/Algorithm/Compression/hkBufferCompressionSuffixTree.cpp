/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

/*

 Compression via suffix tree construction and matching.
 
 Backreferences into the uncompressed stream are found by constructing suffix trees representing
 the input data. A series of suffix trees are created to cover overlapping segments of the input.
 At each point, the two or three suffix trees which intersect the current 8KB lookup window are
 queried.



 The suffix trees are constructed by Ukkonen's algorithm [1], and the sliding-window lookup 
 algorithm is described in [2].

 [1] On-line construction of suffix trees, 
		Esko Ukkonen
		Algorithmica, 1995

 [2] Linear Algorithm for Data Compression via String Matching, 
		Michael Rodeh, Vaughan R. Pratt, Shimon Even, 
		Journal of the ACM (JACM), 1981
 */


#include <Common/Base/hkBase.h>
#include <Common/Base/Algorithm/Compression/hkBufferCompression.h>
#include <Common/Base/Algorithm/Compression/hkBufferCompressionInternal.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

namespace hkBufferCompression
{

/*
 To save space, strings in the suffix tree are represented as (left,right) pointers.
 We use [first, last+1] pairs rather than [first, last] (as in Ukkonen) to make the
 address calculations later involve fewer +/- 1 offsets.
*/
class Substring
{
	public:

		enum {STR_INFINITY = 32767};
		HK_FORCE_INLINE int size()
		{
			return m_end - m_start;
		}
		HK_FORCE_INLINE int size(int maxlen)
		{
			return (m_end == STR_INFINITY ? maxlen : m_end) - m_start;
		}
		HK_FORCE_INLINE void moveTo(int newstart)
		{
			if (m_end != STR_INFINITY)
			{
				m_end = static_cast<short>(newstart + size());
			}
			m_start = static_cast<short>(newstart);
		}
		HK_FORCE_INLINE void offset(short delta)
		{
			moveTo(m_start + delta);
		}

		HK_FORCE_INLINE Substring(short l, short r) : m_start(l), m_end(r)
		{
		}

		HK_FORCE_INLINE Substring(short l) : m_start(l), m_end(STR_INFINITY)
		{
		}

		HK_FORCE_INLINE Substring() : m_start(0), m_end(0)
		{
		}

		short m_start;
		short m_end;
};


class STree
{
	public:

		STree(int maxlen)
			: m_nodes(maxlen * 2 + 10) // each char of input can produce 2 nodes, and there are a few more for root, endmarker, etc.
		{
			reset(HK_NULL);
		}

		void advance(int n)
		{
			for (int i=0;i<n;i++)
			{
				advance();
			}
		}

		/* 
		  Consume another character of input and update the suffix tree.
		  This involves the methods update(), canonize(), and test_split(),
		  whose workings are fully described in the Ukkonen paper.

		  Complexity: amortized O(1)
		 */
		void advance()
		{
			if (!m_root)
			{
				//first hkUchar
				m_root = newNode(0, Substring(0));
				m_active = m_root;
				m_active_str = Substring(1,1);
				m_maxlen = 0;
			}
			else
			{
				update(m_data[m_active_str.m_end]);
				m_active_str.m_end++;
				canonize();
			}
			m_maxlen++;
		}
		void finish()
		{
			if (m_root)
			{
				update(Node::ENDMARKER);
				m_active = 0;
				m_active_str = Substring(0,0);
			}
		}
		void reset(const hkUchar* data)
		{
			m_root = 0;
			m_data = data;
			m_nodes.clear();
		}

	#if 0
		void dump()
		{
			if (m_root)
			{
				m_nodes.get(m_root)->dump(m_nodes, m_maxlen, m_data);
			}
			printf("\n");
		}
	#endif

private:
		inline int maxNodes() const;
		class Node;
		friend class LeanForwardExplicitStack;
public:
		struct LeanForwardExplicitStack
		{
			enum State_
			{ 
				FIRST_VISIT, 
				DOING_NEXT,  // node is waiting for next
				DOING_LEFT	 // node has both a left and a right, left is being analysed and node is waiting for right
			};
			typedef hkEnum<State_, hkUchar> State;
			Node** node;
			short* spos;
			State* state;
			int len;
			LeanForwardExplicitStack(const STree& t)
			{
				//make these bigger than they'll need to be
				len = t.maxNodes();
				node = hkMemTempBufAlloc<Node*>(len);
				spos = hkMemTempBufAlloc<short>(len);
				state = hkMemTempBufAlloc<State>(len);
			}
			~LeanForwardExplicitStack()
			{
				hkMemTempBufFree(node, len);
				hkMemTempBufFree(spos, len);
				hkMemTempBufFree(state, len);
			}
		};

		/*
		"Lean the tree forward". This is necessary to use match<RightmostMatch>, see the comments
		in that method for an explanation.

		Complexity: O(n)
		*/
		void leanForward(LeanForwardExplicitStack& stk)
		{
			if (m_root)
			{
				//m_nodes.get(m_root)->leanForward(m_nodes);
				Node::leanForwardNoRecursion(m_nodes, m_nodes.get(m_root), stk, m_data);
			}
		}



		struct LeftmostMatch
		{
			bool operator()(int a, int b)
			{
				return a >= b;
			}
		};
		struct RightmostMatch
		{
			bool operator()(int a, int b)
			{
				return a < b;
			}
		};

		/* 
		   Search for the longest matching prefix of a search string in the suffix tree.
		   Depending on the template parameter, this algorithm will either search for the
		   longest match starting before the given position or the longest match starting
		   after the given position.
		
		   Direction == LeftmostMatch
			 Search for the longest match whose first hkUcharacter is in [0,startpos)
			 If multiple equally-sized matches are found, returns the leftmost one.
		   Direction == RightmostMatch
			 Search for the longest match whose first hkUcharacter is in [startpos,inf)
			 If multiple equally-sized matches are found, returns the rightmost one.
	      
		   LeftmostMatch requires a left-leaning tree and RightmostMatch a right-leaning one.
		   Trees generated by Ukkonen's algorithm are already left-leaning, so no setup
		   is required to use match<LeftmostMatch>. However, to use RightmostMatch, leanForward()
		   must first be called.
		*/
		template <class Direction>
		Substring match(int startpos, const hkUchar* search, int length) const
		{
			Direction cmp;
			Substring bestmatch(0,0);
			int offset = 0;
			const hkUchar* search_end = search+length;
			Node* n;
			for (NodeID nid = m_root; nid != 0 && search < search_end; nid = n->m_next)
			{
				n = m_nodes.get(nid)->findChild(m_nodes, search[0]);
				if (!n)
				{
					break;
				}

				if (cmp(n->m_str.m_start, startpos + offset))
				{
					//it's only valid if n.start comes before [after] the target in the data stream
					//since the tree is left-leaning [right-leaning], if this node is invalid all children
					//will be invalid also, so we can early-out here.
					break;
				}

				int l = hkMath::min2(n->m_str.size(m_maxlen), int(search_end - search));
				// we can skip the first character, we know it must match since how else
				// could we have found this node?
				HK_ASSERT(0x5435AB43, search[0] == m_data[n->m_str.m_start]);
				int matchcount = 1;
				for (; matchcount < l; matchcount++)
				{
					if (m_data[n->m_str.m_start + matchcount] != search[matchcount])
					{
						break;
					}
				}

				// We have a match at n.start through n.start + matchcount
				// We know it's the longest and therefore best so far
				// since we're visiting these in order of increasing length
				bestmatch.m_start = static_cast<short>(n->m_str.m_start - offset);
				bestmatch.m_end = static_cast<short>(n->m_str.m_start + matchcount);

				if (matchcount < l)
				{
					// partial match so no subsequent matches are valid
					break;
				}

				search += matchcount;
				offset += matchcount;
			}
			return bestmatch;
		}

	private:
		/*
		 The suffix tree is represented as a ternary search tree. Each branch node of the suffix tree
		 has at least one child. If a node has more than one child, the set of sibling children are held
		 in a binary search tree maintained in the m_left and m_right fields. Leaf nodes store no 
		 information and are represented by null pointers.
		 */
		typedef short NodeID;
		class Node
		{
			public:
				typedef unsigned short KeyType;
				enum { ENDMARKER = 0xffff };
				/*
				  To save space, node pointers are represented as 16-bit indices into a buffer of nodes
				  managed by this class.
				 */
				class Allocator
				{
					public:
						Allocator (int arraylen)
							: m_arraylen(arraylen), m_nodearray(hkMemTempBufAlloc<Node>(m_arraylen))
						{
							clear();
						}
						~Allocator ()
						{
							hkMemTempBufFree(m_nodearray, m_arraylen);
						}

						void clear()
						{
							m_maxalloc = 1;
						}

						int maxNodes() const
						{
							return m_arraylen;
						}

						HK_FORCE_INLINE Node* get(NodeID n) const
						{
							HK_ASSERT(0x4325b3f2, n > 0 && n < m_maxalloc);
							return &m_nodearray[n];
						}

						HK_FORCE_INLINE NodeID toID(Node* x) const
						{
							HK_ASSERT(0xfa42b343, x > m_nodearray && x < m_nodearray + m_maxalloc);
							return static_cast<NodeID>(x - m_nodearray);
						}

						HK_FORCE_INLINE NodeID make(Node::KeyType k, NodeID next, const Substring& str)
						{
							HK_ASSERT(0x543ba342, m_maxalloc < m_arraylen);

							NodeID id = static_cast<NodeID>(m_maxalloc++);
							Node* n = get(id);
							n->m_key = k;
							n->m_next = next;
							n->m_str = str;
							n->m_left = 0;
							n->m_right = 0;

							return id;
						}
					private:
						int m_arraylen;
						Node* const m_nodearray;
						int m_maxalloc;
				};


				HK_FORCE_INLINE Node* findChild(const Allocator& nodes, KeyType k)
				{
					NodeID next = nodes.toID(this);
					do
					{
						Node* n = nodes.get(next);
						if (n->m_key == k)
						{
							return n;
						}
						if (n->m_key < k)
						{
							next = n->m_right;
						}
						else
						{
							next = n->m_left;
						}
					} while (next);
					return HK_NULL;
				}

				HK_FORCE_INLINE void addChild(const Allocator& nodes, NodeID newnode)
				{
					Node* p = this;
					KeyType k = nodes.get(newnode)->m_key;
					while (1)
					{
						if (p->m_key < k)
						{
							if (p->m_right)
							{
								p = nodes.get(p->m_right);
							}
							else
							{
								p->m_right = newnode;
								break;
							}
						}
						else
						{
							if (p->m_left)
							{
								p = nodes.get(p->m_left);
							}
							else
							{
								p->m_left = newnode;
								break;
							}
						}
					}
				}


#if 0
				void dump(const Allocator& nodes, int mlen, hkUchar* data, int indent = 0){
					if (m_left)nodes.get(m_left)->dump(nodes, mlen, data, indent);
					printf("%.*s%c: %.*s (%d:%d)\n",indent,"                                                ",m_key==ENDMARKER?'$':(hkUchar)m_key,m_str.size(mlen),data+m_str.m_start,m_str.m_start,m_str.size(mlen));
					if (m_next)nodes.get(m_next)->dump(nodes, mlen, data,indent+1);
					if (m_right)nodes.get(m_right)->dump(nodes, mlen, data, indent);
				}
#endif

				/*
				  To "lean the tree forward", i.e. prepare it for rightmost searches via match<RightmostMatch>,
				  we find the maximal allowable value for each substring on an edge of the tree by doing a post-
				  order traversal of the nodes and setting the parent node's substring to be the max of all of
				  the found positions.
				 */
#if 0
				int leanForward(const Allocator& nodes)
				{
					// Concise and elegant and stack-overflowing version of this algorithm
					int spos = m_str.m_start;
					if (m_next)
					{
						int s = nodes.get(m_next)->leanForward(nodes) - m_str.size();
						if (s > spos)
						{
							spos = s;
							m_str.moveTo(s);
						}
					}
					if (m_left)
					{
						spos = hkMath::max2(spos, nodes.get(m_left)->leanForward(nodes));
					}
					if (m_right)
					{
						spos = hkMath::max2(spos, nodes.get(m_right)->leanForward(nodes));
					}
					return spos;
				}
#endif


				static void leanForwardNoRecursion(const Allocator& nodes, Node* root, LeanForwardExplicitStack& stk, const hkUchar* data)
				{
					/*
					The algorithm is as follows:
					int Node.leanForward(){
					int s = next->leanForward() - str.size();
					if (s > str.start) str.moveTo(s)
					return max{s, str.start, left->leanForward(), right->leanForward()}
					}
					Unfortunately, it must be implemented non-recursively with an explicit
					stack to avoid stack overflow.
					*/

					typedef LeanForwardExplicitStack Stack;
					int top = 0;

					Node* node = root;
					Stack::State state = Stack::FIRST_VISIT;
					short spos = -1; // this will be initialised when the root is visited below

					while (1)
					{
						switch(state)
						{
						case Stack::FIRST_VISIT:
							// just found this node
							if (node->m_next)
							{
								// we have to visit the next node then come back
								stk.node[top] = node;
								stk.state[top] = Stack::DOING_NEXT;
								stk.spos[top] = spos;
								top++;

								node = nodes.get(node->m_next);
								state = Stack::FIRST_VISIT;
								spos = node->m_str.m_start;
								break;
							}
							else
							{
								state = Stack::DOING_NEXT;
								stk.spos[top] = spos;
								// fall through to the next state
							}

						case Stack::DOING_NEXT:
							// we recursed to follow the next pointer of this node and we've returned
							{
								// here spos is the max of the children's positions and stk.spos[top] is
								// the running max of sibling's positions.
								if (node->m_next)
								{
									int s = spos - node->m_str.size();
									HK_ASSERT(0x796ac957, data[s] == data[node->m_str.m_start]);
									if (s > node->m_str.m_start)
									{
										node->m_str.moveTo(s);
									}
								}

								spos = hkMath::max2(node->m_str.m_start, stk.spos[top]);
								if (node->m_left && node->m_right)
								{
									// we have to visit both left and right so we'll have to come back
									stk.node[top] = node;
									stk.state[top] = Stack::DOING_LEFT;
									top++;

									node = nodes.get(node->m_left);
									state = Stack::FIRST_VISIT;
								}
								else if (node->m_left || node->m_right)
								{
									// we only have to visit either left or right, so we can tail-recurse
									NodeID tovisit = node->m_left ? node->m_left : node->m_right;
									node = nodes.get(tovisit);
									state = Stack::FIRST_VISIT;
								}
								else
								{
									// no siblings, return
									if (top == 0)
									{
										return; // recursion is finished, quit loop
									}
									else
									{
										top--;
										node = stk.node[top];
										state = stk.state[top];
									}
								}
							}
							break;

						case Stack::DOING_LEFT:
							//we've just looked at node->m_left, we tail-recurse to node->m_right then we're done
							//node->m_right must exist since otherwise we wouldn't end up in this state
							node = nodes.get(node->m_right);
							state = Stack::FIRST_VISIT;
							break;
						}
					}
				}

			private:
				KeyType m_key;
				NodeID m_left;
				NodeID m_right;

			public:
				NodeID m_next;
				Substring m_str;

				NodeID m_suffix;
		};





		HK_FORCE_INLINE NodeID newNode(Node::KeyType k, NodeID next, const Substring& str)
		{
			return m_nodes.make(k, next, str);
		}
		HK_FORCE_INLINE NodeID newNode(NodeID next, const Substring& str)
		{
			return m_nodes.make(m_data[str.m_start], next, str);
		}


		HK_FORCE_INLINE void canonize()
		{
			NodeID active = m_active;
			int start = m_active_str.m_start;
			int end = m_active_str.m_end;
			while (start != end)
			{
				Node* next = m_nodes.get(active)->findChild(m_nodes, m_data[start]);
				if (next->m_str.size() > (end - start))
				{
					break;
				}
				start += next->m_str.size();
				active = next->m_next;
			}
			m_active = active;
			m_active_str.m_start = static_cast<short>(start);
		}

		HK_FORCE_INLINE NodeID test_split(Node::KeyType t)
		{
			if (m_active_str.size() != 0)
			{
				// implicit state, split
				hkUchar c = m_data[m_active_str.m_start];
				Node* curr = m_nodes.get(m_active)->findChild(m_nodes, c);
				short breakpoint = static_cast<short>(curr->m_str.m_start + m_active_str.size());
				if (Node::KeyType(m_data[breakpoint]) == t)
				{
					return 0;
				}

				// curr -> next becomes curr -> r -> next

				NodeID r = newNode(curr->m_next, Substring(breakpoint, curr->m_str.m_end));

				curr->m_next = r;
				curr->m_str.m_end = breakpoint;
				return r;
			}
			else
			{
				// explicit state
				if (m_nodes.get(m_active)->findChild(m_nodes, t))
				{
					// already exists, so this is the end-point
					return 0;
				}
				else
				{
					// no need to split
					return m_active;
				}
			}
		}




		HK_FORCE_INLINE void update(Node::KeyType newchar)
		{
			NodeID oldr = m_root;
			while (1)
			{
				NodeID r = test_split(newchar);
				if (!r)
				{
					break;
				}
				m_nodes.get(oldr)->m_suffix = r;
				oldr = r;
				m_nodes.get(r)->addChild(m_nodes, newNode(newchar, 0, Substring(m_active_str.m_end)));

				if (m_active == m_root)
				{
					if (m_active_str.size() == 0)
					{
						m_active_str.m_start++;
						break;
					}
					else
					{
						m_active_str.m_start++;
					}
				}
				else
				{
					m_active = m_nodes.get(m_active)->m_suffix;
				}
				canonize();
			}

			m_nodes.get(oldr)->m_suffix = m_active;

		}


	private:
		Node::Allocator m_nodes;

		NodeID m_active;
		Substring m_active_str;

		NodeID m_root;

		const hkUchar* m_data;

		int m_maxlen;
};
// forward declaration rules prevent this from being defined above
int STree::maxNodes() const
{
	return m_nodes.maxNodes();
}

/*
 Compression via a sliding window on overlapped suffix trees, similar to
 [2].
          i                j
		  |				   |
 |----------------|		   |					prev. suffix tree
          |  |----------------|					curr. suffix tree
		  |              |----------------|		next. suffix tree
          |				   |
		  |----------------|					window
  
 The string to be matched starts at position j. We are searching for the 
 longest match of a prefix of the target string inside the window.

 We search for the rightmost match in the prev. tree right of position i
 (via STree::match<RightmostMatch>), the leftmost match in the curr. tree
 left of position j (via STree::match<LeftmostMatch>), and the leftmost
 match in the next tree left of position j.

 In theory, we should be looking for the rightmost match in the current
 tree, but taking the leftmost makes linear-time implementation easier.
 We know that the entirety of the current tree fits inside the window,
 so this is valid.

 Sometimes, the window does not intersect the next tree so we only search
 prev and curr.

 The size of the suffix trees is equal to the length of the window, and the
 size of the overlap is equal to the length of the longest allowable match
 (since we can't find matches that span two trees).
*/
static hk_size_t _suffixTreeCompress(const hkUchar* data, hk_size_t length, hkUchar* outbuf, hk_size_t outlength)
{
	// Compress by matching against a series of suffix trees
	// of overlapping portions of the data.
	// Each tree covers MAXOFF bytes, with an overlap of MAXLONGMATCH
	const int WINDOWSIZE = MAX_MATCH_OFFSET;
	const int OVERLAP = MAX_LONG_MATCH_LEN;
	const int WINDOWSTEP = WINDOWSIZE - OVERLAP;
	HK_COMPILE_TIME_ASSERT(MAX_MATCH_OFFSET >= ((2*MAX_LONG_MATCH_LEN)-1) );

	const hkUchar* data_end = data + length;

	int window_start = 0;

	STree tree1(WINDOWSIZE), tree2(WINDOWSIZE), tree3(WINDOWSIZE);

	STree::LeanForwardExplicitStack leanForwardStack(tree1);

	STree* curr = &tree1;
	STree* prev = &tree2;
	STree* next = &tree3;

	const hkUchar* d = data;
	curr->reset(data);
	curr->advance(hkMath::min2(WINDOWSIZE, (int)length));
	curr->finish();

	CompressedOutput out(outbuf);
	while (d < data_end && out.bytesWritten() < outlength - 10)
	{
		const hkUchar* window_data = data+window_start;
		next->reset(window_data + WINDOWSTEP);
		next->advance(hkMath::max2(0, hkMath::min2( int(WINDOWSIZE), int(data_end - (window_data + WINDOWSTEP)))));
		next->finish();

		int sz = hkMath::min2(WINDOWSIZE, static_cast<int>(length - window_start));
		while (d<window_data + sz && out.bytesWritten() < outlength - 10)
		{
			int dlen = (int)(hkMath::min2(d + OVERLAP, data_end) - d);
			
			Substring m = curr->match<STree::LeftmostMatch>((int)(d - window_data), d, dlen);
			
			if (d >= window_data + WINDOWSTEP)
			{
				//check next window because of overlap
				Substring m2 = next->match<STree::LeftmostMatch>((int)(d - window_data) - WINDOWSTEP, d, dlen);
				if (m2.size() >= m.size()) // nonstrict compare gives precedence to next over curr
				{
					//m2 is relative to the next window, convert it into current window coordinates
					m2.offset(WINDOWSTEP);
					m = m2;
				}
			}

			{
				Substring m2 = prev->match<STree::RightmostMatch>((int)(d - window_data) - OVERLAP, d, dlen);
				if (m2.size() > m.size()) // strict compare gives precedence to curr/next over prev
				{
					// m2 is relative to the previous window, convert it into current window coordinates
					m2.offset(-WINDOWSTEP);
					m = m2;
				}
			}

			if (m.size() >= 3)
			{
				// match
				if (m.size() > MAX_LONG_MATCH_LEN)
				{
					// match is too long, shorten it
					m.m_end = (short)(m.m_start + MAX_LONG_MATCH_LEN);
				}
				// convert the match into absolute data coordinates
				// we need to use ints because the Substring class can't hold such large numbers
				const hkUchar* match_start = data + window_start + static_cast<int>(m.m_start);
				
				out.writeBackref(m.size(), (int)(d - match_start));
				d += m.size();
			}
			else
			{
				out.writeLiteral(*d);
				d++;	
			}
		}

		window_start += WINDOWSTEP;
		// shift down one place
		hkAlgorithm::swap(prev, curr);
		hkAlgorithm::swap(curr, next);
		prev->leanForward(leanForwardStack);
	}
	out.endLiteralRun();
	if (out.bytesWritten() >= outlength-10)
	{
		return 0;
	}
	else
	{
		return out.bytesWritten();
	}
}

} // namespace hkBufferCompression

hk_size_t HK_CALL hkBufferCompression::suffixTreeCompress(const void* data, hk_size_t length, void* outbuf, hk_size_t outlength)
{
	return _suffixTreeCompress( static_cast<const hkUchar*>(data), length, static_cast<hkUchar*>(outbuf), outlength);
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
