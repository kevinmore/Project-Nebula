/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>
#if !defined(HK_PLATFORM_SPU)
#	include <Common/Base/Container/LocalArray/hkLocalArray.h>
#endif

#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>

#include <Physics2012/Collide/BroadPhase/hkpBroadPhaseHandle.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhaseHandlePair.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhaseCastCollector.h>
#include <Physics2012/Collide/BroadPhase/3AxisSweep/hkp3AxisSweep.h>

#if defined HK_ENABLE_DETERMINISM_CHECKS
#	include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#endif

#ifdef HK_PLATFORM_SPU

#include <Common/Base/Spu/Config/hkSpuConfig.h>
#include <Physics2012/Collide/Query/Multithreaded/Spu/hkpSpuConfig.h>
#include <Common/Base/Memory/PlatformUtils/Spu/SpuDmaCache/hkSpu4WayCache.h>
#include <Common/Base/Spu/Dma/Iterator/hkSpuReadOnlyIterator.h>

#endif

#include <Common/Base/Config/hkOptionalComponent.h>

#if defined HK_COMPILER_MSVC
	// C4100 conversion from 'int' to 'unsigned short', possible loss of data
#	pragma warning(disable: 4244)
#endif

#ifdef HK_ARCH_X64
#  undef  HK_COMPILER_HAS_INTRINSICS_IA32
#endif

// ppc variable shifts are microcoded. Use LUT instead.
#if defined(HK_ARCH_PPC)
static const hkUint32 g_bitsLookup[32] = { 1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6,
	1<<7, 1<<8,	1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15, 1<<16, 1<<17, 1<<18,
	1<<19, 1<<20, 1<<21, 1<<22, 1<<23, 1<<24, 1<<25, 1<<26, 1<<27, 1<<28, 1<<29,
	1<<30, 1u<<31};
#	define BIT_SHIFT_LEFT_1( AMOUNT ) (g_bitsLookup[AMOUNT])
#	define BIT_SHIFT_LEFT( VALUE, AMOUNT ) ((VALUE)*g_bitsLookup[AMOUNT])
#else
#	define BIT_SHIFT_LEFT_1( AMOUNT ) (1<<(AMOUNT))
#	define BIT_SHIFT_LEFT( VALUE, AMOUNT ) ((VALUE)<<(AMOUNT))
#endif

static inline void HK_CALL memclear16( void *dest, int bytesize )
{
	// warning: this function clears only the previous-multiple-of-16 bytes !
	bytesize >>= 4;
#if defined(HK_REAL_IS_DOUBLE)
	hkUint32* d = (hkUint32*)dest;
	while ( bytesize >=0 )
	{
		*(d++) = 0;
		*(d++) = 0;
		*(d++) = 0;
		*(d++) = 0;
		bytesize--;
	}
#else
	hkVector4 val; val.setZero();
	hkVector4 *d = (hkVector4 *)dest;
	while ( bytesize >=0 )
	{
		*(d++) = val;
		bytesize--;
	}
#endif
}

hkpBroadPhase::BroadPhaseType hkp3AxisSweep::getType() const
{
#if !defined(HK_BROADPHASE_32BIT)
	return BROADPHASE_16BIT;
#else
	return BROADPHASE_32BIT;
#endif
}

static inline void  staticFlipBit( hkUint32* bitField, int index )
{
	int pos = index>>5;
	int bit = BIT_SHIFT_LEFT_1(index & 0x1f);
	bitField[pos] ^= bit;
}

static inline void staticClearBit( hkUint32* bitField, int index )
{
	int pos = index>>5;
	int bit = BIT_SHIFT_LEFT_1(index & 0x1f);
	bitField[pos] &= ~bit;
}

static inline int staticIsBitSet( const hkUint32* bitField, int index )
{
	int pos = index>>5;
	int bit = BIT_SHIFT_LEFT_1(index & 0x1f);
	return bitField[pos] & bit;
}

static inline void HK_CALL memset4( void *dest, int value, int bytesize )
{
	int *d = (int *)dest;
	bytesize >>= 2;
	while ( bytesize >=0 )
	{
		*(d++) = value;
		bytesize--;
	}
}

// define to ensure some functions are expanded inline
#if defined(HK_DEBUG) || 1
#	define HK_CHECK_FOR_INLINED_USING_CONST( NEVER_TRUE_EXPR )
#else
#	define HK_CHECK_FOR_INLINED_USING_CONST( NEVER_TRUE_EXPR ) \
		if( NEVER_TRUE_EXPR ) \
		{ \
			extern void this_function_was_not_inlined(); \
			this_function_was_not_inlined(); \
		}
#endif


//inline void this_function_was_not_inlined(){	int x = 0; }



//
//	The maximum extents of the broadphase in int space: we cannot use the full
//  16 bit as we have to use the sign bit (without the overflow bit) to compare
//  two values;
//

#if defined(HK_PLATFORM_GC) && defined(HK_COMPILER_MWERKS)
	hkUint32 hkp3AxisSweep::OneMask[4] = { 0x00000001, 0x00000001, 0x00000001, 0x00000001};
#else
	HK_ALIGN16( hkUint32 hkp3AxisSweep::OneMask[4] ) = { 0x00000001, 0x00000001, 0x00000001, 0x00000001};
#endif
hkQuadReal hkp3AxisSweep::MaxVal = HK_QUADREAL_CONSTANT( 
	hkReal(hkp3AxisSweep::AABB_MAX_FVALUE) , hkReal(hkp3AxisSweep::AABB_MAX_FVALUE),
	hkReal(hkp3AxisSweep::AABB_MAX_FVALUE) , hkReal(hkp3AxisSweep::AABB_MAX_FVALUE)
	);

int hkp3AxisSweep::getNumMarkers() const
{
					return m_numMarkers;
}



#define NP (static_cast<hkp3AxisSweep::hkpBpNode*>(0))
const int hkp3AxisSweep::hkpBpNode::s_memberOffsets[] = {
	hkGetByteOffsetInt( &NP->min_y, &NP->min_x),
	hkGetByteOffsetInt( &NP->min_y, &NP->max_x),
	hkGetByteOffsetInt( &NP->min_y, &NP->min_y),
	hkGetByteOffsetInt( &NP->min_y, &NP->max_y),
	hkGetByteOffsetInt( &NP->min_y, &NP->min_z),
	hkGetByteOffsetInt( &NP->min_y, &NP->max_z)
};
#undef NP 


void hkp3AxisSweep::_convertAabbToInt( const hkAabb& aabb, hkAabbUint32& aabbOut ) const
{
	HK_ASSERT2(0xaf542fe1, m_scale.length<4>().getReal() != 0, "Make sure to call set32BitOffsetAndScale() after creating the broadphase.");

	_convertAabbToInt( aabb, m_offsetLow, m_offsetHigh, m_scale, aabbOut );
}


void hkp3AxisSweep::convertVectorToInt( const hkVector4& vec, hkUint32* intsOut) const
{
	HK_ASSERT2(0xaf542fe7, m_scale.length<4>().getReal() != 0, "Make sure to call set32BitOffsetAndScale() after creating the broadphase.");

	hkVector4 clipMin; clipMin.setZero();
	hkVector4 clipMax; clipMax.m_quad = hkp3AxisSweep::MaxVal;

#	ifndef HK_BROADPHASE_32BIT
	HK_ALIGN16(hkIntUnion64 mi);
	hkVector4Util::convertToUint16WithClip( vec, m_offsetLow, m_scale,	clipMin, clipMax, mi );
	intsOut[0] = mi.u16[0];
	intsOut[1] = mi.u16[1];
	intsOut[2] = mi.u16[2];
	intsOut[3] = mi.u16[3];
#	else
	hkVector4Util::convertToUint32WithClip( vec, m_offsetLow, m_scale,	clipMin, clipMax, intsOut );
#	endif
}

hkUint32 hkp3AxisSweep::hkpBpNode::yzDisjoint( const hkpBpNode& other ) const
{
#ifndef HK_BROADPHASE_32BIT
	hkUint32 minA = *reinterpret_cast<const hkUint32*>(&min_y);
	hkUint32 minB = *reinterpret_cast<const hkUint32*>(&other.min_y);
	hkUint32 maxA = *reinterpret_cast<const hkUint32*>(&max_y);
	hkUint32 maxB = *reinterpret_cast<const hkUint32*>(&other.max_y);
	maxA -= minB;
	maxB -= minA;
	maxA |= maxB;		// or the sign bits
	maxA &= 0x80008000;	// get the sign bits
	return maxA;
#else
	hkUint32 a = hkUint32(max_y) - other.min_y;
	hkUint32 b = hkUint32(max_z) - other.min_z;
	hkUint32 c = hkUint32(other.max_y) - min_y;
	hkUint32 d = hkUint32(other.max_z) - min_z;
	a |= b;
	c |= d;
	c |= a;
	return c & 0x80000000;
#endif
}

hkInt32 hkp3AxisSweep::hkpBpNode::xyDisjoint( const hkpBpNode& other ) const
{
	BpInt a = max_y - other.min_y;
	BpInt b = max_x - other.min_x;
	BpInt c = other.max_y - min_y;
	BpInt d = other.max_x - min_x;
	a |= b;
	c |= d;
	c |= a;
#ifndef HK_BROADPHASE_32BIT
	return c & 0x8000;
#else
	return c& 0x80000000;
#endif
}

hkInt32 hkp3AxisSweep::hkpBpNode::xzDisjoint( const hkpBpNode& other ) const
{
#ifndef HK_BROADPHASE_32BIT
	hkInt16 a = max_x - other.min_x;
	hkInt16 b = max_z - other.min_z;
	hkInt16 c = other.max_x - min_x;
	hkInt16 d = other.max_z - min_z;
	a |= b;
	c |= d;
	c |= a;
	return c & 0x8000;
#else
	hkUint32 a = max_x - other.min_x;
	hkUint32 b = max_z - other.min_z;
	hkUint32 c = other.max_x - min_x;
	hkUint32 d = other.max_z - min_z;
	a |= b;
	c |= d;
	c |= a;
	return c & 0x80000000;
#endif
}


void hkp3AxisSweep::beginOverlap( hkpBpNode& a, hkpBpNode& b, hkArray<hkpBroadPhaseHandlePair>& newPairsOut)
{
	hkpBroadPhaseHandlePair& pair = newPairsOut.expandOne();
	pair.m_a = a.m_handle;
	pair.m_b = b.m_handle;
}

void hkp3AxisSweep::endOverlap(   hkpBpNode& a, hkpBpNode& b, hkArray<hkpBroadPhaseHandlePair>& deletedPairsOut)
{
	hkpBroadPhaseHandlePair& pair = deletedPairsOut.expandOne();
	pair.m_a = a.m_handle;
	pair.m_b = b.m_handle;
}

void hkp3AxisSweep::beginOverlapCheckMarker(   hkpBpMarker* markers, hkpBpNode& a, int nodeIndexA, hkpBpNode& b, hkArray<hkpBroadPhaseHandlePair>& newPairsOut)
{
	if ( !b.isMarker() )
	{
		hkpBroadPhaseHandlePair& pair = newPairsOut.expandOne();
		pair.m_a = a.m_handle;
		pair.m_b = b.m_handle;
	}
	else
	{
		hkpBpMarker& m = b.getMarker(markers);
		m.m_overlappingObjects.pushBack(nodeIndexA);
	}
}

void hkp3AxisSweep::endOverlapCheckMarker  (   hkpBpMarker* markers, hkpBpNode& a, int nodeIndexA, hkpBpNode& b, hkArray<hkpBroadPhaseHandlePair>& deletedPairsOut)
{
	if ( !b.isMarker() )
	{
		hkpBroadPhaseHandlePair& pair = deletedPairsOut.expandOne();
		pair.m_a = a.m_handle;
		pair.m_b = b.m_handle;
	}
	else
	{
		hkpBpMarker& m = b.getMarker(markers);
		int i = m.m_overlappingObjects.indexOf( nodeIndexA);
		m.m_overlappingObjects.removeAt( i );
	}
}



#define DISJOINT( axisIndex, node, otherNode )  ( (axisIndex==0 && node.yzDisjoint(otherNode)) || (axisIndex==1 && node.xzDisjoint(otherNode)) || (axisIndex==2 && node.xyDisjoint(otherNode)) )

#if defined( HK_COMPILER_GCC ) && ( defined( HK_PLATFORM_GC ) || defined( HK_PLATFORM_PSP ) || defined( HK_PLATFORM_LINUX ) )
// The ngc sn, PSP(R) (PlayStation(R)Portable) sn, and linux gcc compilers cannot compile the templated
// function node._getMin, so this alternative has been implemented as a
// replacement.  Unfortunately, it is not specific to any version of gcc as
// 3.1.0 works, but 3.3.6 does not.
template<int index>
HK_FORCE_INLINE hkp3AxisSweep::BpInt& _getNodeXYZ( hkp3AxisSweep::BpInt& val_x, hkp3AxisSweep::BpInt& val_y, hkp3AxisSweep::BpInt& val_z )
{
	if ( index == 0 ) return val_x;
	if ( index == 1 ) return val_y;
	return val_z;
}
#define NODE_GET_MIN(node, epIndex) _getNodeXYZ<epIndex>( node.min_x, node.min_y, node.min_z )
#define NODE_GET_MAX(node, epIndex) _getNodeXYZ<epIndex>( node.max_x, node.max_y, node.max_z )
#else
#define NODE_GET_MIN(node, epIndex) node._getMin< epIndex >()
#define NODE_GET_MAX(node, epIndex) node._getMax< epIndex >()
#endif

template<int axisIndex, hkp3AxisSweep::hkpBpMarkerUse marker>
HK_FORCE_INLINE void _updateAxis( hkp3AxisSweep* sweep, hkp3AxisSweep::hkpBpNode* nodes, hkp3AxisSweep::hkpBpNode& node, hkUint32 nodeIndex, hkUint32 new_min, hkUint32 new_max, hkArray<hkpBroadPhaseHandlePair>& newPairsOut, hkArray<hkpBroadPhaseHandlePair>& deletedPairsOut)
{
	HK_CHECK_FOR_INLINED_USING_CONST( axisIndex == 5 );

	hkp3AxisSweep::hkpBpAxis& axis = sweep->m_axis[axisIndex];

	//HK_TIMER_BEGIN_LIST("update", "0");
	{
		// move min to left
		hkUint32 minEpIndex = NODE_GET_MIN(node, axisIndex);
		hkp3AxisSweep::hkpBpEndPoint* ep = &axis.m_endPoints[minEpIndex];
		hkUint32 otherValue;
		while ( new_min < (otherValue = ep[-1].m_value) )
		{
			hkUint32 otherNodeIndex = ep[-1].m_nodeIndex; // 16 -> 32
			hkp3AxisSweep::hkpBpNode& otherNode = nodes[otherNodeIndex];

			ep[0] = ep[-1];
			ep--;

			if ( hkp3AxisSweep::hkpBpEndPoint::isMaxPoint(otherValue))
			{
				hkInt32 disjoint;

				if ( axisIndex == 0 )		{			disjoint = node.yzDisjoint(otherNode);			}
				else if ( axisIndex == 1 )	{			disjoint = node.xzDisjoint(otherNode);			}
				else						{			disjoint = node.xyDisjoint(otherNode);			}

				NODE_GET_MAX(otherNode, axisIndex) = minEpIndex;// assign after disjoint test so that no LHS on Xbox360 (as vals are 16bit to same 32bit)


				if (!disjoint)
				{
					if ( marker == hkp3AxisSweep::HK_BP_NO_MARKER)
					{
						sweep->beginOverlap( node, otherNode, newPairsOut );
					}
					else
					{
						sweep->beginOverlapCheckMarker( sweep->m_markers, node, nodeIndex, otherNode, newPairsOut );
					}
				}
			}
			else
			{
				NODE_GET_MIN(otherNode, axisIndex) = minEpIndex;
			}
			minEpIndex--;
		}

#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
		{
			hkUint32 otherNodeIndex;

			while ( new_min == ep[-1].m_value && nodeIndex < (otherNodeIndex = ep[-1].m_nodeIndex) )
			{
				HK_ASSERT(0x46734956, otherNodeIndex != 0 );
				ep[0] = ep[-1];
				ep--;
				hkp3AxisSweep::hkpBpNode& otherNode = nodes[otherNodeIndex];
				NODE_GET_MIN(otherNode, axisIndex) = minEpIndex;
				minEpIndex--;
			}
		}
#endif 
		ep->m_nodeIndex = nodeIndex; // 32 -> 16
		ep->m_value     = new_min;
		NODE_GET_MIN(node, axisIndex) = minEpIndex;
	}


	//HK_TIMER_SPLIT_LIST("1");
	{
		// move max to right

		hkUint32 maxEpIndex = NODE_GET_MAX(node, axisIndex);
		hkp3AxisSweep::hkpBpEndPoint* maxEp = &axis.m_endPoints[maxEpIndex];
		{
			hkUint32 otherValue;
			while ( new_max > (otherValue = maxEp[1].m_value) )
			{
				maxEp[0] = maxEp[1];
				maxEp++;
				maxEpIndex++;

				hkp3AxisSweep::hkpBpNode& otherNode = nodes[maxEp->m_nodeIndex];

				if ( !hkp3AxisSweep::hkpBpEndPoint::isMaxPoint(otherValue))
				{
					hkInt32 disjoint;

					if ( axisIndex == 0 )		{			disjoint = node.yzDisjoint(otherNode);			}
					else if ( axisIndex == 1 )	{			disjoint = node.xzDisjoint(otherNode);			}
					else						{			disjoint = node.xyDisjoint(otherNode);			}

					NODE_GET_MIN(otherNode, axisIndex)--;	// assign after disjoint test so that no LHS on Xbox360 (as vals are 16bit, so in same 32bit word)

					if (!disjoint)
					{
						if ( marker == hkp3AxisSweep::HK_BP_NO_MARKER)
						{
							sweep->beginOverlap( node, otherNode, newPairsOut );
						}
						else
						{
							sweep->beginOverlapCheckMarker( sweep->m_markers, node, nodeIndex, otherNode, newPairsOut );
						}
					}
				}
				else
				{
					NODE_GET_MAX(otherNode, axisIndex)--;
				}
			}
#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
			{
				hkUint32 otherNodeIndex;

				while ( new_max == maxEp[1].m_value && nodeIndex > (otherNodeIndex = maxEp[1].m_nodeIndex) && otherNodeIndex != 0)
				{
					maxEp[0] = maxEp[1];
					maxEp++;
					maxEpIndex++;
					HK_ASSERT(0x24e89c94, otherNodeIndex != 0 );

					hkp3AxisSweep::hkpBpNode& otherNode = nodes[otherNodeIndex];
					NODE_GET_MAX(otherNode, axisIndex)--;
				}
			}
#endif 
		}

		{
			//	HK_TIMER_SPLIT_LIST("2");
			// move max to left
			hkUint32 otherValue;
			while ( new_max < (otherValue = maxEp[-1].m_value) )
			{
				maxEp[0] = maxEp[-1];
				hkp3AxisSweep::hkpBpNode& otherNode = nodes[maxEp[-1].m_nodeIndex];
				maxEp--;
				maxEpIndex--;
				if ( !hkp3AxisSweep::hkpBpEndPoint::isMaxPoint(otherValue))
				{
					hkInt32 disjoint;

					if ( axisIndex == 0 )		{			disjoint = node.yzDisjoint(otherNode);			}
					else if ( axisIndex == 1 )	{			disjoint = node.xzDisjoint(otherNode);			}
					else						{			disjoint = node.xyDisjoint(otherNode);			}

					NODE_GET_MIN(otherNode, axisIndex)++;

					if (!disjoint)
					{
						if ( marker == hkp3AxisSweep::HK_BP_NO_MARKER)
						{
							sweep->endOverlap( node, otherNode, deletedPairsOut );
						}
						else
						{
							sweep->endOverlapCheckMarker( sweep->m_markers, node, nodeIndex, otherNode, deletedPairsOut );
						}
					}
				}
				else
				{
					NODE_GET_MAX(otherNode,axisIndex)++;
				}
			}

#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
			{
				hkUint32 otherNodeIndex;

				while ( new_max == maxEp[-1].m_value && nodeIndex < (otherNodeIndex = maxEp[-1].m_nodeIndex) )
				{
					maxEp[0] = maxEp[-1];
					maxEp--;
					maxEpIndex--;
					hkp3AxisSweep::hkpBpNode& otherNode = nodes[otherNodeIndex];
					NODE_GET_MAX(otherNode, axisIndex)++;
				}
			}
#endif 
		}
		NODE_GET_MAX(node, axisIndex) = maxEpIndex;
		maxEp->m_value		= new_max;
		maxEp->m_nodeIndex = nodeIndex;
	}

	//	HK_TIMER_SPLIT_LIST("3");
	{
		// move min to right
		hkUint32 minEpIndex = NODE_GET_MIN(node, axisIndex);
		hkp3AxisSweep::hkpBpEndPoint* ep = &axis.m_endPoints[minEpIndex];
		hkUint32 otherValue;
		while ( new_min > ( otherValue = ep[1].m_value) )
		{
			hkp3AxisSweep::hkpBpNode& otherNode = nodes[ep[1].m_nodeIndex];
			ep[0] = ep[1];

			ep++;
			minEpIndex++;

			if ( hkp3AxisSweep::hkpBpEndPoint::isMaxPoint(otherValue))
			{
				hkInt32 disjoint;

				if ( axisIndex == 0 )		{			disjoint = node.yzDisjoint(otherNode);			}
				else if ( axisIndex == 1 )	{			disjoint = node.xzDisjoint(otherNode);			}
				else						{			disjoint = node.xyDisjoint(otherNode);			}

				NODE_GET_MAX(otherNode, axisIndex)--;

				if (!disjoint)
				{
					if ( marker == hkp3AxisSweep::HK_BP_NO_MARKER)
					{
						sweep->endOverlap( node, otherNode, deletedPairsOut );
					}
					else
					{
						sweep->endOverlapCheckMarker( sweep->m_markers, node, nodeIndex, otherNode, deletedPairsOut );
					}
				}
			}
			else
			{
				NODE_GET_MIN(otherNode, axisIndex)--;
			}
		}

#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
		{
			hkUint32 otherNodeIndex;

			while ( new_min == ep[1].m_value && nodeIndex > (otherNodeIndex = ep[1].m_nodeIndex) )
			{
				ep[0] = ep[1];
				ep++;
				minEpIndex++;
				HK_ASSERT(0x69b392b8, otherNodeIndex != 0 );

				hkp3AxisSweep::hkpBpNode& otherNode = nodes[otherNodeIndex];
				NODE_GET_MIN(otherNode, axisIndex)--;
			}
		}
#endif 

		NODE_GET_MIN(node, axisIndex) = minEpIndex;
		ep->m_nodeIndex = nodeIndex;
		ep->m_value		= new_min;
	}
	//	HK_TIMER_END_LIST();
}

template<int axisIndex>
HK_FORCE_INLINE void _fixDeterministicAxisOrderAfterNodeIdWasDecreased( hkp3AxisSweep* sweep, hkp3AxisSweep::hkpBpNode* nodes, hkp3AxisSweep::hkpBpNode& node, hkUint32 nodeIndex )
{
	hkp3AxisSweep::hkpBpAxis& axis = sweep->m_axis[axisIndex];

	{
		// move min to left
		hkUint32 minEpIndex = NODE_GET_MIN(node, axisIndex);
		hkp3AxisSweep::hkpBpEndPoint* ep = &axis.m_endPoints[minEpIndex];
		{
			hkUint32 otherNodeIndex;

			while ( ep[0].m_value == ep[-1].m_value && nodeIndex < (otherNodeIndex = ep[-1].m_nodeIndex) )
			{
				ep[0] = ep[-1];
				ep--;
				hkp3AxisSweep::hkpBpNode& otherNode = nodes[otherNodeIndex];
				NODE_GET_MIN(otherNode, axisIndex) = minEpIndex;
				minEpIndex--;
			}
		}
		ep->m_nodeIndex = nodeIndex; // 32 -> 16
		NODE_GET_MIN(node, axisIndex) = minEpIndex;
	}

	{
		hkUint32 maxEpIndex = NODE_GET_MAX(node, axisIndex);
		hkp3AxisSweep::hkpBpEndPoint* maxEp = &axis.m_endPoints[maxEpIndex];
		{
			// move max to left
			{
				hkUint32 otherNodeIndex;
				while ( maxEp[0].m_value == maxEp[-1].m_value && nodeIndex < (otherNodeIndex = maxEp[-1].m_nodeIndex) )
				{
					maxEp[0] = maxEp[-1];
					maxEp--;
					hkp3AxisSweep::hkpBpNode& otherNode = nodes[otherNodeIndex];
					NODE_GET_MAX(otherNode, axisIndex) = maxEpIndex;
					maxEpIndex--;
				}
			}
		}
		maxEp->m_nodeIndex = nodeIndex;
		NODE_GET_MAX(node, axisIndex) = maxEpIndex;
	}
	//	HK_TIMER_END_LIST();
}

void hkp3AxisSweep::_fixDeterministicOrderAfterNodeIdWasDecreased( int nodeIndex )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	hkpBpNode* nodes = &m_nodes[0];
	hkpBpNode& node = m_nodes[nodeIndex];


	_fixDeterministicAxisOrderAfterNodeIdWasDecreased<0>( this, nodes, node, nodeIndex  );
	_fixDeterministicAxisOrderAfterNodeIdWasDecreased<1>( this, nodes, node, nodeIndex  );
	_fixDeterministicAxisOrderAfterNodeIdWasDecreased<2>( this, nodes, node, nodeIndex  );
}

void hkp3AxisSweep::updateAabb( hkpBroadPhaseHandle* object, const hkAabbUint32& aabb, hkArray<hkpBroadPhaseHandlePair>& newPairsOut, hkArray<hkpBroadPhaseHandlePair>& deletedPairsOut)
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	//CHECK_CONSISTENCY();

	hkpBpNode* nodes = &m_nodes[0];
	hkUlong nodeIndex = (hkUlong)(object->m_id);
	hkpBpNode& node = nodes[ nodeIndex ];

	_updateAxis<0,HK_BP_USE_MARKER>( this, nodes, node, nodeIndex, aabb.m_min[0], aabb.m_max[0], newPairsOut, deletedPairsOut );
	_updateAxis<1,HK_BP_NO_MARKER>( this, nodes, node, nodeIndex, aabb.m_min[1], aabb.m_max[1], newPairsOut, deletedPairsOut );
	_updateAxis<2,HK_BP_NO_MARKER>( this, nodes, node, nodeIndex, aabb.m_min[2], aabb.m_max[2], newPairsOut, deletedPairsOut );

	CHECK_CONSISTENCY();
}


#if !defined(HK_PLATFORM_SPU)
static hkBool pairCmpLess( hkpBroadPhaseHandlePair& pairA, hkpBroadPhaseHandlePair& pairB )
{
	return ( pairA.m_a->m_id < pairB.m_a->m_id ) ||
		   ( pairA.m_a->m_id == pairB.m_a->m_id && pairA.m_b->m_id < pairB.m_b->m_id );
}

void hkp3AxisSweep::updateAabbs( hkpBroadPhaseHandle* objects[], const hkAabb* aabbs, int numObjects, hkArray<hkpBroadPhaseHandlePair>& addedPairsOut, hkArray<hkpBroadPhaseHandlePair>& removedPairsOut )
{
	CHECK_CONSISTENCY();

	hkpBroadPhaseHandle** oEnd = objects + numObjects;

	for ( hkpBroadPhaseHandle** o = &objects[0]; o < oEnd; o++)
	{
		hkAabbUint32 aabb;
		_convertAabbToInt( *aabbs, aabb );

#	if defined (HK_ENABLE_DETERMINISM_CHECKS)
		hkCheckDeterminismUtil::checkMtCrc( 0xf00001d0, aabb.m_min, 3);
		hkCheckDeterminismUtil::checkMtCrc( 0xf00001d0, aabb.m_max, 3);
#endif
		updateAabb( *o, aabb, addedPairsOut, removedPairsOut );

		aabbs++;
	}

	CHECK_CONSISTENCY();
}
#endif


void hkp3AxisSweep::updateAabbsUint32( hkpBroadPhaseHandle* objects[], const hkAabbUint32* aabbs, int numObjects, hkArray<hkpBroadPhaseHandlePair>& addedPairsOut, hkArray<hkpBroadPhaseHandlePair>& removedPairsOut )
{
	CHECK_CONSISTENCY();

	hkpBroadPhaseHandle** oEnd = objects + numObjects;

	for ( hkpBroadPhaseHandle** o = &objects[0]; o < oEnd; o++)
	{
		hkAabbUint32 aabb;
		convertAabbToBroadPhaseResolution(*aabbs, aabb);

#	if defined (HK_ENABLE_DETERMINISM_CHECKS)
		hkCheckDeterminismUtil::checkMtCrc( 0xf00001df, aabb.m_min, 3);
		hkCheckDeterminismUtil::checkMtCrc( 0xf00001df, aabb.m_max, 3);
#endif
		updateAabb( *o, aabb, addedPairsOut, removedPairsOut );

		aabbs++;
	}
	CHECK_CONSISTENCY();
}

#if defined(HK_BROADPHASE_32BIT)
	static const int broadPhaseFlags = hkpBroadPhase::ISA_SWEEP_AND_PRUNE;
	static const hkpBroadPhase::BroadPhaseType broadPhaseType = hkpBroadPhase::BROADPHASE_32BIT;
#else
	static const int broadPhaseFlags = hkpBroadPhase::ISA_SWEEP_AND_PRUNE | hkpBroadPhase::SUPPORT_SPU_RAY_CAST | hkpBroadPhase::SUPPORT_SPU_LINEAR_CAST | hkpBroadPhase::SUPPORT_SPU_CLOSEST_POINTS | hkpBroadPhase::SUPPORT_SPU_CHAR_PROXY_INT;
	static const hkpBroadPhase::BroadPhaseType broadPhaseType = hkpBroadPhase::BROADPHASE_16BIT;
#endif


#if !defined(HK_PLATFORM_SPU)
hkp3AxisSweep::hkp3AxisSweep(const hkVector4& worldMin, const hkVector4& worldMax, int numMarkers)
	:	hkpBroadPhase(broadPhaseType, sizeof(*this), broadPhaseFlags)
{
	m_aabb.m_min = worldMin;
	m_aabb.m_max = worldMax;
	HK_ASSERT( 0xf0452134, numMarkers <= 256 );

	if( !numMarkers)
	{
		numMarkers++;
	}
	int ldNumMarkers = -1;
	int x = numMarkers;
	while ( x > 0)
	{
		ldNumMarkers++;
		x = x>>1;
	}

	if ( (1<<(ldNumMarkers)) != numMarkers)
	{
		HK_WARN(0x5036c744,  "The number of markers has to be 0,1,2,4,16...256");
	}


	const int estimatesMaxNumObjects = 255;
	m_nodes.reserve(estimatesMaxNumObjects);

	//
	//	setup m_intToFloatFloorCorrection
	//
	{
		m_offsetLow.setZero();
		m_offsetHigh.setZero();
		m_scale.setAll( 1.0f );

		hkReal l = 10.0f;
		hkReal h = 11.0f;
		hkAabb aabb;
		hkAabbUint32 aabbInt;
		for (int i = 0; i < 23; i++ )
		{
			hkReal m = (l+h) * 0.5f;
			aabb.m_min.setAll( m );
			aabb.m_max.setAll( m + 1 );
			_convertAabbToInt( aabb, aabbInt );

			if ( aabbInt.m_max[0] < 12 )
			{
				l = m;
			}
			else
			{
				h = m;
			}
		}
		m_intToFloatFloorCorrection = (l+h) * 0.5f - 11;
	}

	m_scale.setZero();
	m_offsetLow.setZero();
	m_offsetHigh.setZero();

	//
	// create a object containing the entire world and insert begin part
	//
	hkpBpNode& node0 = m_nodes.expandOne();
	{
		// Initialize the min's and handle.
		// The max's get set further down.
		node0.min_x = node0.min_y = node0.min_z = 0;
		node0.m_handle = HK_NULL;

		const int maxEndPoints = estimatesMaxNumObjects * 2 + 2;
		const int nMarkers = (1<<ldNumMarkers)-1;

		m_axis[0].m_endPoints.reserve(maxEndPoints + 2 * nMarkers);
		m_axis[1].m_endPoints.reserve(maxEndPoints);
		m_axis[2].m_endPoints.reserve(maxEndPoints);


		hkpBpEndPoint ep;
		ep.m_nodeIndex = 0;
		ep.m_value = AABB_MIN_VALUE & (~1);
		m_axis[0].m_endPoints.pushBackUnchecked(ep);
		m_axis[1].m_endPoints.pushBackUnchecked(ep);
		m_axis[2].m_endPoints.pushBackUnchecked(ep);
	}

	//
	//	setup markers
	//
	{
		m_ld2NumMarkers = ldNumMarkers;
		m_numMarkers = (1<<m_ld2NumMarkers)-1;

		m_markers = HK_NULL;
		if (m_numMarkers)
		{
			m_markers = hkAllocate<hkpBpMarker>( m_numMarkers, HK_MEMORY_CLASS_BROAD_PHASE );
		}

		for (int i = 0; i < m_numMarkers; i++)
		{
			hkpBpMarker& marker = *new ( &m_markers[i] ) hkpBpMarker();

			hkpBpEndPoint ep;

			ep.m_nodeIndex = m_nodes.getSize();
			ep.m_value = (i+1)<<(HK_BP_NUM_VALUE_BITS-m_ld2NumMarkers);
			marker.m_nodeIndex = ep.m_nodeIndex;
			marker.m_value = ep.m_value;

			hkpBpNode& node = m_nodes.expandOne();

			node.min_x = m_axis[0].m_endPoints.getSize();
			m_axis[0].m_endPoints.pushBackUnchecked(ep);

			ep.m_value |= 1;
			node.max_x = m_axis[0].m_endPoints.getSize();
			m_axis[0].m_endPoints.pushBackUnchecked(ep);

			node.m_handle = reinterpret_cast<hkpBroadPhaseHandle*>( 1 | (i * sizeof(hkpBpMarker)));
			node.min_y = node.min_z = 0;
			node.max_y = node.max_z = 1;
		}
	}

	//
	// insert end part of node0
	//
	{
		node0 = m_nodes[0]; // reset the reference in case m_nodes got resized.
		node0.max_x = m_axis[0].m_endPoints.getSize();
		node0.max_y = m_axis[1].m_endPoints.getSize();
		node0.max_z = m_axis[2].m_endPoints.getSize();

		hkpBpEndPoint ep;
		ep.m_nodeIndex = 0;
		ep.m_value = BpInt(AABB_MAX_VALUE | 1);

		m_axis[0].m_endPoints.pushBackUnchecked(ep);
		m_axis[1].m_endPoints.pushBackUnchecked(ep);
		m_axis[2].m_endPoints.pushBackUnchecked(ep);
	}

	CHECK_CONSISTENCY();
}
#endif


#if !defined(HK_PLATFORM_SPU)
hkp3AxisSweep::~hkp3AxisSweep()
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	for (int i = 0; i < getNumMarkers(); i++)
	{
		m_markers[i].~hkpBpMarker();
	}
	hkDeallocate<hkpBpMarker>(m_markers);
}
#endif


#if !defined(HK_PLATFORM_SPU)

#if defined(HK_BROADPHASE_32BIT)
HK_COMPILE_TIME_ASSERT( (int)hkp3AxisSweep::AABB_MAX_FVALUE == (int)hkAabbUtil::AABB_UINT32_MAX_FVALUE );
#endif

void hkp3AxisSweep::set32BitOffsetAndScale(const hkVector4& offsetLow, const hkVector4& offsetHigh, const hkVector4& scale)
{
	// Use the supplied 32bit values as is.
	m_offsetLow32bit  = offsetLow;
	m_offsetHigh32bit = offsetHigh;
	m_scale32bit      = scale;

#if defined(HK_BROADPHASE_32BIT)

	// Use the supplied 32bit values as is.
	m_offsetLow  = offsetLow;
	m_offsetHigh = offsetHigh;
	m_scale      = scale;

#else

	// Calculate the 16bit values.
	m_offsetLow = offsetLow;

	hkVector4 span,spanInv;
	span.setSub( m_aabb.m_max, m_aabb.m_min );

#if defined(HK_PLATFORM_IOS) && defined(__llvm__)
	
	
	
	__asm__("nop"); // Work around llvm bug
#endif
	
	hkReal maxSpan = span.horizontalMax<3>().getReal();
	if ( maxSpan > 12000.0f)
	{
		HK_WARN( 0xf034de45, "Your broadphase extents is bigger than 12k meters, this can lead to a performance penalty, try to reduce the broadphase size or use the 32 bit broadphase" );
	}

	hkVector4 rounding;
	rounding.setMul( hkSimdReal::fromFloat(1.0f/hkReal(AABB_MAX_FVALUE)), span );
	m_offsetHigh.setAdd(m_offsetLow, rounding);

	spanInv.setReciprocal(span);
	m_scale.setMul( hkSimdReal::fromFloat(hkReal(AABB_MAX_FVALUE)), spanInv );

	m_scale     .zeroComponent<3>();
	m_offsetLow .zeroComponent<3>();
	m_offsetHigh.zeroComponent<3>();

#endif
}
#endif


#if !defined(HK_PLATFORM_SPU)
	// endpoint array is now layed out so new sorted elements start at indexOfNewEps
	// we merge them in starting from the
	// beginning of the list, so we can update node backpointers.
	// We use the space newly swapped in endpoints were at to hold endpoints being shifted
	// when one of these temporarily help endpoints is swapped back in we update the backpointer
	// merges a batch, scratch has to be of size indexOfNewEps
void hkp3AxisSweep::hkpBpAxis::mergeBatch( hkpBpNode *nodes, int indexOfNewEps, int newNum, int axis,  hkpBpEndPoint* scratch)
{
	// now we merge
	hkpBpEndPoint* oldList = scratch;
	{
		for( int i = 0; i < indexOfNewEps; i++ )
		{
			oldList[i] = m_endPoints[i];
		}
	}

	// start at index 1 to skip the guard at zero
	// only merged gets written to, so it's safe to declare HK_RESTRICT
	hkpBpEndPoint* merged   = &m_endPoints[1];
	const hkpBpEndPoint *mergeOld = &oldList[1];
	const hkpBpEndPoint *mergeNew = &m_endPoints[indexOfNewEps];

		// The end of the list: Note: we do not merge the last element
	const hkpBpEndPoint *oldEnd = &oldList[0] + indexOfNewEps -1;

	const hkpBpEndPoint *newEnd = &m_endPoints[0] + (indexOfNewEps+newNum);

	// skip elements not modified
	while( mergeOld->m_value < mergeNew->m_value )
	{
		mergeOld++;
		merged++;
	}

	if( mergeOld < oldEnd && mergeNew < newEnd )
	{
		while(1)
		{
			if( mergeOld->m_value < mergeNew->m_value 
#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
					|| (mergeOld->m_value == mergeNew->m_value && mergeOld->m_nodeIndex < mergeNew->m_nodeIndex )
#endif
			)
			{
				*merged = *mergeOld;
				nodes[ merged->m_nodeIndex ].setElem( axis, merged->isMaxPoint(), merged - &m_endPoints[0] );
				mergeOld++;
				merged++;
				if ( mergeOld >=  oldEnd )
				{
					break;
				}
			}
			else
			{
				*merged = *mergeNew;
				nodes[ merged->m_nodeIndex ].setElem( axis, merged->isMaxPoint(), merged - &m_endPoints[0] );
				mergeNew++;
				merged++;
				if ( mergeNew >=  newEnd )
				{
					break;
				}
			}
		}
	}

	while( mergeNew < newEnd )
	{
		*merged = *mergeNew;

		nodes[ merged->m_nodeIndex ].setElem( axis, merged->isMaxPoint(), merged - &m_endPoints[0] );

		mergeNew++;
		merged++;
	}

		// copy the old rest including the last element
	while( mergeOld <= oldEnd )
	{
		*merged = *mergeOld;

		nodes[ merged->m_nodeIndex ].setElem( axis, merged->isMaxPoint(), merged - &m_endPoints[0] );

		mergeOld++;
		merged++;
	}

}
#endif


#if !defined(HK_PLATFORM_SPU)
template<int axis>
void hkp3AxisSweep::hkpBpAxis::removeBatch( hkpBpNode* nodes, const hkArrayBase<int>& nodeRelocations )
{
	const hkpBpEndPoint* epEnd =  m_endPoints.end();
	hkpBpEndPoint* epw = m_endPoints.begin();	// end point to write
	int iw = 0;									// write index
	for ( hkpBpEndPoint* epr = m_endPoints.begin(); epr < epEnd; epr++ )
	{
		int nodeIndex = epr->m_nodeIndex;
		int newIndex = nodeRelocations[nodeIndex];
		if ( newIndex < 0 )
		{
			// delete node
			continue;
		}

		// relink node
		*epw = *epr;
		epw->m_nodeIndex = newIndex;
		nodes[ newIndex ].setElem( axis, epr->isMaxPoint(), iw );

		epw++;
		iw++;
	}
	m_endPoints.setSize( iw );
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::hkpBpAxis::insert( hkpBpNode *nodes, int nodeIndex, BpInt minPosition, BpInt maxPosition, BpInt& minInsertPositionOut, BpInt&  maxInsertPositionOut )
{
	int newSize = m_endPoints.getSize() + 2;
	m_endPoints.setSize( newSize );

	hkpBpEndPoint *ep = &m_endPoints[newSize - 3];

	// always move the guard node
	{
		ep[2] = ep[0];
		ep--;
	}

	// first search the max value elements by two
	while ( maxPosition < ep->m_value )
	{
		ep[2] = ep[0];
		ep--;
	}

#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
	while ( maxPosition == ep->m_value && (hkUint32)nodeIndex < ep->m_nodeIndex )
	{
		ep[2] = ep[0];
		ep--;
	}
#endif

	ep[2].m_nodeIndex = nodeIndex;
	ep[2].m_value = maxPosition;
	maxInsertPositionOut = ep - &m_endPoints[0] + 2;

	while ( minPosition < ep->m_value )
	{
		ep[1] = ep[0];
		ep--;
	}
#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
	while ( minPosition == ep->m_value && (hkUint32)nodeIndex < ep->m_nodeIndex )
	{
		ep[1] = ep[0];
		ep--;
	}
#endif

	ep[1].m_value = minPosition;
	ep[1].m_nodeIndex = nodeIndex;
	minInsertPositionOut = ep - &m_endPoints[0] + 1;
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::hkpBpAxis::remove( int minPosition, int maxPosition )
{
	hkpBpEndPoint *ep = &m_endPoints[minPosition];
	hkpBpEndPoint *epEnd = &m_endPoints[maxPosition]-1;
	while ( ep < epEnd )
	{
		ep[0] = ep[1];
		ep++;
	}

	epEnd = &m_endPoints[m_endPoints.getSize()-2];
	m_endPoints.setSizeUnchecked( m_endPoints.getSize()-2 );
	while ( ep < epEnd )
	{
		ep[0] = ep[2];
		ep++;
	}
}
#endif


#if !defined(HK_PLATFORM_SPU)
static HK_FORCE_INLINE hkp3AxisSweep::BpInt HK_CALL calcOffset( hkp3AxisSweep::BpInt x, hkp3AxisSweep::BpInt mi, hkp3AxisSweep::BpInt ma )
{
	hkUint32 x32 = x;
	hkUint32 mi32 = mi;
	hkUint32 ma32 = ma;
	int dist = (unsigned(mi32 - x32) >> 31) + (unsigned(ma32 - x32) >> 31);
	return dist;
}

static HK_FORCE_INLINE hkp3AxisSweep::BpInt HK_CALL calcOffset2( hkp3AxisSweep::BpInt x, hkp3AxisSweep::BpInt mi, hkp3AxisSweep::BpInt shift )
{
	hkUint32 x32 = x;
	hkUint32 mi32 = mi;
	int dist = (unsigned(mi32 - x32) >> 31)  & shift;
	return dist;
}
#endif


#if !defined(HK_PLATFORM_SPU)

#if defined(HK_COMPILER_MSVC)
#	pragma warning(disable: 4799)
#endif


#if HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED && defined(HK_PLATFORM_PS3_PPU) && !defined(HK_BROADPHASE_32BIT)
// update all the indices pointing into the m_axis arrays to reflect an insert
void hkp3AxisSweep::updateNodesAfterInsert( hkpBpNode *nodes, int numNodes, hkpBpNode& newNode )
{
	//
	//	we have to add +1 if the old_index > newNode.min - 1
	//  we have to add +1 if the old_index > newNode.max - 2
	//
	// what we do is to subtract and use the sign bit as an add

	vec_ushort8 mi, ma;
	{
		const vec_uchar16 minPerm = (vec_uchar16){ 0,1,2,3, 0,1,2,3, 8,9,8,9,		12,13,14,15};
		const vec_uchar16 maxPerm = (vec_uchar16){ 4,5,6,7, 4,5,6,7, 10,11,10,11,	12,13,14,15};

		const vec_ushort8 m0 = *reinterpret_cast<const vec_ushort8*>(&newNode);
		
		vec_ushort8 one = vec_splat_u16(1);
		vec_ushort8 two = vec_splat_u16(2);

		mi = vec_perm(m0, m0, minPerm);
		ma = vec_perm(m0, m0, maxPerm);
		mi = vec_sub(mi, one);
		ma = vec_sub(ma, two);
	}

	vec_ushort8* node = reinterpret_cast<vec_ushort8*>(nodes);
	vec_ushort8* endNode = node + numNodes;
	vec_ushort8 protectPointerMask = (vec_ushort8){ 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0x0000, 0x0000};

	for (; node < endNode; node++)
	{	
		vec_ushort8 m = node[0];
		vec_ushort8 micmp = (vec_ushort8)vec_cmpgt( m, mi );
		vec_ushort8 macmp = (vec_ushort8)vec_cmpgt( m, ma );
		vec_ushort8 negAdd = vec_add( micmp, macmp );
		vec_ushort8 toSub = vec_and( negAdd, protectPointerMask);
		node[0] = vec_sub( m, toSub );
	}
}
#else
// update all the indices pointing into the m_axis arrays to reflect an insert
void hkp3AxisSweep::updateNodesAfterInsert( hkpBpNode *nodes, int numNodes, hkpBpNode& newNode )
{
	//
	//  we have to add +1 if the old_index > newNode.min - 1
	//  we have to add +1 if the old_index > newNode.max - 2
	//
	//  what we do is to subtract and use the sign bit as an add

	HK_ALIGN16( hkpBpNode miNode );
	HK_ALIGN16( hkpBpNode maNode );
	miNode.min_x = newNode.min_x - 1;
	miNode.max_x = newNode.min_x - 1;
	miNode.min_y = newNode.min_y - 1;
	miNode.max_y = newNode.min_y - 1;
	miNode.min_z = newNode.min_z - 1;
	miNode.max_z = newNode.min_z - 1;

	maNode.min_x = newNode.max_x - 2;
	maNode.max_x = newNode.max_x - 2;
	maNode.min_y = newNode.max_y - 2;
	maNode.max_y = newNode.max_y - 2;
	maNode.min_z = newNode.max_z - 2;
	maNode.max_z = newNode.max_z - 2;

#if HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED && defined(HK_COMPILER_HAS_INTRINSICS_IA32)&& !defined(HK_BROADPHASE_32BIT)
	const __m64 miYZ = (__m64&)miNode.min_y;
	const __m64 maYZ = (__m64&)maNode.min_y;
	const __m64 miX  = (__m64&)miNode.min_x;
	const __m64 maX  = (__m64&)maNode.min_x;
	const __m64 xmask = _mm_set_pi32(0, 0xffffffff);

	__m64 *node = reinterpret_cast<__m64*>(nodes);
	__m64 *endNode = node + 2 * numNodes;
	for (; node < endNode; node+=2)
	{
		__m64 YZ = node[0];
		__m64 X  = node[1];

		__m64 addYZ = _mm_add_pi16(_mm_cmpgt_pi16(YZ, miYZ), _mm_cmpgt_pi16(YZ, maYZ) );
		__m64 addX0 = _mm_add_pi16(_mm_cmpgt_pi16(X,  miX) , _mm_cmpgt_pi16(X,  maX) );

		__m64 addX  = _mm_and_si64(xmask, addX0);

		node[0] = _mm_sub_pi16( YZ, addYZ );
		node[1] = _mm_sub_pi16( X,  addX );
	}

#elif HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED && defined(HK_PLATFORM_XBOX360) && !defined(HK_BROADPHASE_32BIT)

	__vector4* node = reinterpret_cast<__vector4*>(nodes);
	__vector4* endNode = node + numNodes;
	HK_ALIGN16( const hkUint32 _protectPointerMask[4] ) = { 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000 };
	__vector4 protectPointerMask = (__vector4&)_protectPointerMask;
	__vector4 mi = (__vector4&)miNode.min_y;
	__vector4 ma = (__vector4&)maNode.min_y;

	for (; node < endNode; node++)
	{	
		__vector4 m = node[0];
		__vector4 micmp = __vcmpgtuh( m, mi );
		__vector4 macmp = __vcmpgtuh( m, ma );
		__vector4 negAdd = __vadduhm( micmp, macmp );
		__vector4 toSub = __vand( negAdd, protectPointerMask );
		node[0] = __vsubuhm( m, toSub );
	}


#else
 	hkpBpNode *node = nodes;
 	hkpBpNode *endNode = nodes + numNodes;

 	for (; node < endNode; node++)
 	{
		node->min_x += calcOffset( node->min_x, miNode.min_x, maNode.min_x );
		node->min_y += calcOffset( node->min_y, miNode.min_y, maNode.min_y );
		node->max_x += calcOffset( node->max_x, miNode.max_x, maNode.max_x );
		node->max_y += calcOffset( node->max_y, miNode.max_y, maNode.max_y );
		node->min_z += calcOffset( node->min_z, miNode.min_z, maNode.min_z );
		node->max_z += calcOffset( node->max_z, miNode.max_z, maNode.max_z );
	}
#endif

	hkVector4Util::exitMmx(); // reset FPU after MMX is used
}
#endif // #if HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED && defined(HK_PLATFORM_PS3_PPU) && !defined(HK_BROADPHASE_32BIT)
#endif //!defined(HK_PLATFORM_SPU)


template<int flip>
HK_FORCE_INLINE void hkp3AxisSweep_appendPair(const hkp3AxisSweep::hkpBpNode& aabb0, const hkp3AxisSweep::hkpBpNode& aabb1, hkp3AxisSweep::hkpBpMarker* markers, hkp3AxisSweep::MarkerHandling markerHandlingForNodesB, hkArray<hkpBroadPhaseHandlePair>& pairs_out)
{
	if ( !flip )
	{
		if ( aabb1.isMarker() )
		{
			if ( markerHandlingForNodesB == hkp3AxisSweep::MARKERS_IGNORE )
			{
				return;
			}

			int i0 = aabb0.m_handle->m_id;	// expensive cache miss here, but the problem is that the input node array might be a copy
			hkp3AxisSweep::hkpBpMarker& m = aabb1.getMarker( markers );
			if ( markerHandlingForNodesB == hkp3AxisSweep::MARKERS_ADD_NEW_OVERLAPS )
			{
				m.m_overlappingObjects.pushBack(i0);
			}
			else
			{
				int i = m.m_overlappingObjects.indexOf( i0 );
				m.m_overlappingObjects.removeAt(i);
			}
			return;
		}
		hkpBroadPhaseHandlePair& pairOut = pairs_out.expandOne();
		pairOut.m_a = aabb0.m_handle;
		pairOut.m_b = aabb1.m_handle;
	}
	else
	{
		if ( aabb0.isMarker() )
		{
			if ( markerHandlingForNodesB == hkp3AxisSweep::MARKERS_IGNORE )
			{
				return;
			}

			int i0 = aabb1.m_handle->m_id;	// expensive cache miss here
			hkp3AxisSweep::hkpBpMarker& m = aabb0.getMarker( markers );
			if ( markerHandlingForNodesB == hkp3AxisSweep::MARKERS_ADD_NEW_OVERLAPS )
			{
				m.m_overlappingObjects.pushBack(i0);
			}
			else
			{
				int i = m.m_overlappingObjects.indexOf( i0 );
				m.m_overlappingObjects.removeAt(i);
			}
			return;
		}
		hkpBroadPhaseHandlePair& pairOut = pairs_out.expandOne();
		pairOut.m_a = aabb1.m_handle;
		pairOut.m_b = aabb0.m_handle;
	}
}

template<int flipKeys>
HK_FORCE_INLINE void hkp3AxisSweep_scanList(	const hkp3AxisSweep::hkpBpNode& query,  const hkp3AxisSweep::hkpBpNode* HK_RESTRICT sxyz, hkp3AxisSweep::hkpBpMarker* markers, hkp3AxisSweep::MarkerHandling markerHandlingForNodesB, hkArray<hkpBroadPhaseHandlePair>& pairs_out )
{
	hkUint32 maxX = query.max_x;
	while( sxyz->min_x < maxX )
	{
		int ov0 = query.yzDisjoint( sxyz[0] );
		if ( !ov0 )
		{
			hkp3AxisSweep_appendPair<flipKeys>( query, sxyz[0], markers, markerHandlingForNodesB, pairs_out );
		}
		sxyz++;
	}
}


void hkp3AxisSweep::collide1Axis( const hkpBpNode* pa, int numA, const hkpBpNode* pb, int numB, MarkerHandling markerHandlingForNodesB, hkArray<hkpBroadPhaseHandlePair>& pairsOut)
{
	HK_ASSERT2(0xad8750aa, numA == 0 || pa[numA-1].min_x != BpInt(-1), "numA should not include the padding elements at the end.");
	HK_ASSERT2(0xad8756aa, numB == 0 || pb[numB-1].min_x != BpInt(-1), "numB should not include the padding elements at the end.");

	HK_ASSERT2(0xad8757aa, pa[numA+0].min_x == BpInt(-1), "One max-value padding elements are required at the end.");
	HK_ASSERT2(0xad8758aa, pb[numB+0].min_x == BpInt(-1), "One max-value padding elements are required at the end.");

#if defined(HK_DEBUG)
	// assert that the input lists are sorted
	{	for (int i =0 ; i < numA-1; i++){ HK_ASSERT( 0xf0341232, pa[i].min_x <= pa[i+1].min_x); }	}
	{	for (int i =0 ; i < numB-1; i++){ HK_ASSERT( 0xf0341233, pb[i].min_x <= pb[i+1].min_x); }	}
#endif
	hkp3AxisSweep::hkpBpMarker* markers = m_markers;
	while ( true )
	{
		if ( pa->min_x > pb->min_x )
		{
			const bool flipKeys = true;
			hkp3AxisSweep_scanList<flipKeys>( *pb, pa, markers, markerHandlingForNodesB, pairsOut );
			pb++;
			if ( --numB <= 0 ) { break; }
		}
		else
		{
			const bool dontflipKeys = false;
			hkp3AxisSweep_scanList<dontflipKeys>( *pa, pb, markers, markerHandlingForNodesB, pairsOut );
			pa++;
			if ( --numA <= 0 ) { break; }
		}
	}
}

void HK_CALL hkp3AxisSweep::collide1Axis( const hkpBpNode* pa, int numA, hkArray<hkpBroadPhaseHandlePair>& pairsOut )
{
	HK_ASSERT2(0xad8751aa, numA == 0 || pa[numA-1].min_x != BpInt(-1), "numA should not include the padding elements at the end.");

	HK_ASSERT2(0xad8757aa, pa[numA+0].min_x == BpInt(-1), "One max-value padding elements are required at the end.");

#if defined(HK_DEBUG)
	// assert that the input lists are sorted
	{	for (int i =0 ; i < numA-1; i++){ HK_ASSERT( 0xf0341232, pa[i].min_x <= pa[i+1].min_x); }	}
#endif

	while ( --numA > 0 )	// this iterates numA-1
	{
		const bool dontflipKeys = false;
		hkp3AxisSweep_scanList<dontflipKeys>( *pa, pa+1, HK_NULL, MARKERS_IGNORE, pairsOut );
		pa++;
	}
}

#ifndef HK_PLATFORM_SPU
// update all the indices pointing into the m_axis arrays to reflect an insert
void hkp3AxisSweep::updateNodesAfterBatchTailInsert( hkpBpNode *nodes, int numNodes, int numNewEndPoints, int* offsets )
{
	//
	//  we have to add +1 if the old_index > newNode.min - 1
	//  we have to add +1 if the old_index > newNode.max - 2
	//
	//  what we do is to subtract and use the sign bit as an add

	HK_ALIGN16( hkpBpNode miNode );
	miNode.min_x = offsets[0] - 1;
	miNode.max_x = offsets[0] - 1;
	miNode.min_y = offsets[1] - 1;
	miNode.max_y = offsets[1] - 1;
	miNode.min_z = offsets[2] - 1;
	miNode.max_z = offsets[2] - 1;


#if HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED && defined(HK_COMPILER_HAS_INTRINSICS_IA32)&& !defined(HK_BROADPHASE_32BIT)
	hkUint32 shift = numNewEndPoints + ( numNewEndPoints << 16 );
	const __m64 miYZ = (__m64&)miNode.min_y;
	const __m64 miX  = (__m64&)miNode.min_x;
	const __m64 yzmask = _mm_set_pi32(shift, shift);
	const __m64 xmask  = _mm_set_pi32(0, shift);

	__m64 *node = reinterpret_cast<__m64*>(nodes);
	__m64 *endNode = node + 2 * numNodes;
	for (; node < endNode; node+=2)
	{
		__m64 YZ = node[0];
		__m64 X  = node[1];

		__m64 addYZ = _mm_and_si64(_mm_cmpgt_pi16(YZ, miYZ), yzmask );
		__m64 addX0 = _mm_and_si64(_mm_cmpgt_pi16(X,  miX) , xmask );

		node[0] = _mm_add_pi16( YZ, addYZ );
		node[1] = _mm_add_pi16( X,  addX0 );
	}

#elif HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED && defined(HK_PLATFORM_PS3_PPU) && !defined(HK_BROADPHASE_32BIT)
	vec_ushort8  mi = (vec_ushort8&)miNode.min_y;

	vec_ushort8* node = reinterpret_cast<vec_ushort8*>(nodes);
	vec_ushort8* endNode = node + numNodes;
	vec_ushort8 protectPointerMask = (vec_ushort8){ numNewEndPoints, numNewEndPoints, numNewEndPoints, numNewEndPoints, numNewEndPoints, numNewEndPoints, 0x0000, 0x0000};

	for (; node < endNode; node++)
	{	
		vec_ushort8 m = node[0];
		vec_ushort8 micmp = (vec_ushort8)vec_cmpgt( m, mi );
		vec_ushort8 toAdd = vec_and( micmp, protectPointerMask);
		node[0] = vec_add( m, toAdd );
	}

#elif HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED && defined(HK_PLATFORM_XBOX360) && !defined(HK_BROADPHASE_32BIT)
	hkUint32 shift = numNewEndPoints + ( numNewEndPoints << 16 );

	__vector4* node = reinterpret_cast<__vector4*>(nodes);
	__vector4* endNode = node + numNodes;
	HK_ALIGN16( const hkUint32 _protectPointerMask[4] ) = { shift, shift, shift, 0x00000000 };
	__vector4 protectPointerMask = (__vector4&)_protectPointerMask;
	__vector4 mi = (__vector4&)miNode.min_y;

	for (; node < endNode; node++)
	{	
		__vector4 m = node[0];
		__vector4 micmp = __vcmpgtuh( m, mi );
		__vector4 toAdd = __vand( micmp, protectPointerMask );
		node[0] = __vadduhm( m, toAdd );
	}
#else
	hkpBpNode *node = nodes;
	hkpBpNode *endNode = nodes + numNodes;

	for (; node < endNode; node++)
	{
		node->min_x += calcOffset2( node->min_x, miNode.min_x, numNewEndPoints );
		node->min_y += calcOffset2( node->min_y, miNode.min_y, numNewEndPoints );
		node->max_x += calcOffset2( node->max_x, miNode.max_x, numNewEndPoints );
		node->max_y += calcOffset2( node->max_y, miNode.max_y, numNewEndPoints );
		node->min_z += calcOffset2( node->min_z, miNode.min_z, numNewEndPoints );
		node->max_z += calcOffset2( node->max_z, miNode.max_z, numNewEndPoints );
	}
#endif

	hkVector4Util::exitMmx(); // reset FPU after MMX is used
}
#endif //!SPU

#if !defined(HK_PLATFORM_SPU)
#if HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED && defined(HK_PLATFORM_PS3_PPU) && !defined(HK_BROADPHASE_32BIT)
void hkp3AxisSweep::updateNodesAfterDelete( hkpBpNode *nodes, int numNodes, hkpBpNode& oldNode )
{
	//
	//	we have to add -1 if the old_index > newNode.min
	//  we have to add -1 if the old_index > newNode.max
	//
	// what we do is to subtract and use the sign bit as an sub

	vec_ushort8 mi, ma;
	{
		const vec_uchar16 minPerm = (vec_uchar16){ 0,1,2,3, 0,1,2,3, 8,9,8,9,		12,13,14,15};
		const vec_uchar16 maxPerm = (vec_uchar16){ 4,5,6,7, 4,5,6,7, 10,11,10,11,	12,13,14,15};

		const vec_ushort8 mNew = *reinterpret_cast<const vec_ushort8*>(&oldNode);
		mi = vec_perm(mNew, mNew, minPerm);
		ma = vec_perm(mNew, mNew, maxPerm);
	}

	vec_ushort8* node = reinterpret_cast<vec_ushort8*>(nodes);
	vec_ushort8* endNode = node + numNodes;
	vec_ushort8 protectPointerMask = (vec_ushort8){ 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0x0000, 0x0000};

	for (; node < endNode; node++)
	{	
		vec_ushort8 m = node[0];

		vec_ushort8 micmp = (vec_ushort8)vec_cmpgt( m, mi );
		vec_ushort8 macmp = (vec_ushort8)vec_cmpgt( m, ma );
		vec_ushort8 negAdd = vec_add( micmp, macmp );
		vec_ushort8 toAdd = vec_and( negAdd, protectPointerMask);
		node[0] = vec_add( m, toAdd );
	}
}
#else

// update all the indices pointing into the m_axis arrays to reflect an insert
void hkp3AxisSweep::updateNodesAfterDelete( hkpBpNode *nodes, int numNodes, hkpBpNode& oldNode )
{
	//
	//	we have to add -1 if the old_index > newNode.min
	//  we have to add -1 if the old_index > newNode.max
	//
	// what we do is to subtract and use the sign bit as an sub
	HK_ALIGN16( hkpBpNode miNode );
	HK_ALIGN16( hkpBpNode maNode );

	miNode.min_x = oldNode.min_x;
	miNode.max_x = oldNode.min_x;
	miNode.min_y = oldNode.min_y;
	miNode.max_y = oldNode.min_y;
	miNode.min_z = oldNode.min_z;
	miNode.max_z = oldNode.min_z;
	miNode.m_handle = reinterpret_cast<hkpBroadPhaseHandle*>( ~(hkUlong)0);

	maNode.min_x = oldNode.max_x;
	maNode.max_x = oldNode.max_x;
	maNode.min_y = oldNode.max_y;
	maNode.max_y = oldNode.max_y;
	maNode.min_z = oldNode.max_z;
	maNode.max_z = oldNode.max_z;
	maNode.m_handle = reinterpret_cast<hkpBroadPhaseHandle*>( ~(hkUlong)0);

#if HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED && defined(HK_COMPILER_HAS_INTRINSICS_IA32) && !defined(HK_BROADPHASE_32BIT)

	const __m64 miYZ = (__m64&)miNode.min_y;
	const __m64 maYZ = (__m64&)maNode.min_y;
	const __m64 miX  = (__m64&)miNode.min_x;
	const __m64 maX =  (__m64&)maNode.min_x;
	const __m64 xmask = _mm_set_pi32(0, 0xffffffff);

	__m64* node = reinterpret_cast<__m64*>(nodes);
	__m64* endNode = node + 2 * numNodes;
	for (; node != endNode; node+=2)
	{
		__m64 YZ = node[0];
		__m64 X  = node[1];

		__m64 addYZ = _mm_add_pi16(_mm_cmpgt_pi16(YZ, miYZ), _mm_cmpgt_pi16(YZ, maYZ) );
		__m64 addX0 = _mm_add_pi16(_mm_cmpgt_pi16(X,  miX) , _mm_cmpgt_pi16(X,  maX) );

		__m64 addX  = _mm_and_si64(addX0, xmask);

		node[0] = _mm_add_pi16( YZ, addYZ );
		node[1] = _mm_add_pi16( X,  addX );
	}

#elif HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED && defined(HK_PLATFORM_XBOX360) && !defined(HK_BROADPHASE_32BIT)

	__vector4* node = reinterpret_cast<__vector4*>(nodes);
	__vector4* endNode = node + numNodes;
	HK_ALIGN16( const hkUint32 _protectPointerMask[4] ) = { 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000 };
	__vector4 protectPointerMask = (__vector4&)_protectPointerMask;
	__vector4 mi = (__vector4&)miNode.min_y;
	__vector4 ma = (__vector4&)maNode.min_y;

	for (; node < endNode; node++)
	{	
		__vector4 m = node[0];
		__vector4 micmp = __vcmpgtuh( m, mi );
		__vector4 macmp = __vcmpgtuh( m, ma );
		__vector4 negAdd = __vadduhm( micmp, macmp );
		__vector4 toAdd = __vand( negAdd, protectPointerMask );
		node[0] = __vadduhm( m, toAdd );
	}

#else
 	hkpBpNode *node = nodes;
 	hkpBpNode *endNode = nodes + numNodes;

 	for (; node != endNode; ++node)
 	{
		node->min_x -= calcOffset( node->min_x, miNode.min_x, maNode.min_x );
		node->max_x -= calcOffset( node->max_x, miNode.max_x, maNode.max_x );
		node->min_y -= calcOffset( node->min_y, miNode.min_y, maNode.min_y );
		node->max_y -= calcOffset( node->max_y, miNode.max_y, maNode.max_y );
		node->min_z -= calcOffset( node->min_z, miNode.min_z, maNode.min_z );
		node->max_z -= calcOffset( node->max_z, miNode.max_z, maNode.max_z );
	}
#endif

	hkVector4Util::exitMmx(); // reset FPU after MMX is used
}

#endif // HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED && defined(HK_PLATFORM_PS3_PPU) && !defined(HK_BROADPHASE_32BIT)
#endif //!defined(HK_PLATFORM_SPU)


#if !defined(HK_PLATFORM_SPU)
	// set all bits of objects which could possible overlap an object based on a range on the x-axis
void hkp3AxisSweep::setBitsBasedOnXInterval( int numNodes, int x_value, const hkpBpNode& queryNode, BpInt queryNodeIndex, hkUint32* bitField) const
{
	int minPos = queryNode.min_x;
	int maxPos = queryNode.max_x;

	memclear16( bitField, numNodes>>3 );
	// run through the x axis and set the bits
	const hkpBpEndPoint *ep =  &m_axis[0].m_endPoints[1];	// skip the first point

	if ( getNumMarkers())
	{
		int pos = x_value >> (HK_BP_NUM_VALUE_BITS-m_ld2NumMarkers);
		if ( pos > 0)
		{
			HK_ASSERT(0x6289b7b7,  pos <= m_numMarkers);
			hkpBpMarker* m = &m_markers[pos-1];
			{
				staticFlipBit( bitField, m->m_nodeIndex);
				BpInt* o = m->m_overlappingObjects.begin();
				for (int k = m->m_overlappingObjects.getSize()-1; k>=0; k--)
				{
					// do not find yourself
					if ( *o != queryNodeIndex)
					{
						staticFlipBit( bitField, *o);
					}
					o++;
				}
				const hkpBpNode& markerNode = m_nodes[ m->m_nodeIndex ];
				ep = &m_axis[0].m_endPoints[ markerNode.min_x + 1];

				//
				//	As our markers actually store all objects overlapping them markers extent,
				//  our marker actually stores to many objects. We actually just want to see
				//  the objects overlapping our markers min_x. Therefore we have to undo some
				//  of the bits set
				//
				{
					const hkpBpEndPoint *end = &m_axis[0].m_endPoints[ markerNode.max_x ];
					while ( ep < end )
					{
						if ( !ep->isMaxPoint() )
						{
							staticClearBit( bitField, ep->m_nodeIndex);
						}
						ep++;
					}
				}
				ep = &m_axis[0].m_endPoints[ markerNode.min_x + 1];
			}
		}
	}

	{
		const hkpBpEndPoint *end = &m_axis[0].m_endPoints[ minPos ];
		while ( ep < end )
		{
			staticFlipBit( bitField, ep->m_nodeIndex);
			ep++;
		}
		ep++;	// do not find yourself
		end = &m_axis[0].m_endPoints[ maxPos ];
		while ( ep < end  )
		{
			if ( !ep->isMaxPoint() )
			{
				staticFlipBit( bitField, ep->m_nodeIndex);
			}
			ep++;
		}
	}
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::reQuerySingleObject( const hkpBroadPhaseHandle* object, hkArray<hkpBroadPhaseHandlePair>& pairs_out) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );

	int numNodes = m_nodes.getSize();

	int scratchSize = (numNodes>>5)+8;
	
	hkArray<hkUint32>::Temp scratchArea( scratchSize );
	hkUint32* scratch = scratchArea.begin();
	hkUint32* bitField = scratch;

	const hkUint32 nodeIndex = object->m_id;
	const hkpBpNode& refNode = m_nodes[nodeIndex];		// the node to delete

	setBitsBasedOnXInterval( numNodes, m_axis[0].m_endPoints[refNode.min_x].m_value, refNode, nodeIndex, bitField);

	hkpBroadPhaseHandlePair pair;
	pair.m_a = const_cast<hkpBroadPhaseHandle*>(object);

	const hkpBpNode *node = &m_nodes[0];
	const hkUint32* end = bitField + (m_nodes.getSize() >> 5) + 1;

	for( ; bitField < end; node += 32, bitField++ ) //check 32 at a time
	{
		const hkpBpNode *n = node;

		for ( hkUint32 mask = bitField[0]; mask; )
		{
			if ( 0 == (mask & 0xff) )
			{
				n += 8;
				mask = mask>>8;
				continue;
			}
			if ( ( mask & 0x1 ) && !n->yzDisjoint( refNode ) )
			{
				if ( !n->isMarker() )
				{
					pair.m_b = n->m_handle;
					pairs_out.pushBack( pair );
				}
			}
			n += 1;
			mask = mask>>1;
		}
	}
}

bool hkp3AxisSweep::areAabbsOverlapping( const hkpBroadPhaseHandle* bhA, const hkpBroadPhaseHandle* bhB ) const
{
	const hkUint32 nodeIndexA = bhA->m_id;
	const hkUint32 nodeIndexB = bhB->m_id;
	const hkpBpNode& nodeA = m_nodes[nodeIndexA];	
	const hkpBpNode& nodeB = m_nodes[nodeIndexB];

	if ( nodeA.xyDisjoint( nodeB ) || nodeA.yzDisjoint( nodeB ))
	{
		return false;
	}
	return true;
}

#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::querySingleAabbAddObject( hkpBroadPhaseHandle* object, int newNodeIndex, const hkUint32 *bitField, hkpBpNode& refNode, hkArray<hkpBroadPhaseHandlePair>& pairs_out) const
{
	hkpBroadPhaseHandlePair pair;
	pair.m_a = object;

	const hkpBpNode *node = &m_nodes[0];
	const hkUint32* end = bitField + (m_nodes.getSize() >> 5) + 1;

	for( ; bitField < end; node += 32, bitField++ ) //check 32 at a time
	{
		const hkpBpNode *n = node;

		for ( hkUint32 mask = bitField[0]; mask; )
		{
			if ( 0 == (mask & 0xff) )
			{
				n += 8;
				mask = mask>>8;
				continue;
			}
			if ( ( mask & 0x1 ) && !n->yzDisjoint( refNode ) )
			{
				if ( !n->isMarker() )
				{
					pair.m_b = n->m_handle;
					pairs_out.pushBack( pair );
				}
				else
				{
					n->getMarker( m_markers ).m_overlappingObjects.pushBack( newNodeIndex );
				}
			}
			n += 1;
			mask = mask>>1;
		}
	}
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::addObject( hkpBroadPhaseHandle* object, const hkAabbUint32& aabbIn, hkArray<hkpBroadPhaseHandlePair>& newPairs, bool border )
{
#ifndef HK_BROADPHASE_32BIT
	HK_ASSERT2(0x451af3d0, m_axis[0].m_endPoints.getSize() < 0x7ffe, "Broadphase overflow. Cannot add more than 2^14 objects (including markers). Use 32bit broadphase. See hkBroadphase class documentation.");
#endif
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	CHECK_CONSISTENCY();

	hkAabbUint32 aabb;
	convertAabbToBroadPhaseResolution(aabbIn, aabb);

	// Border objects should be snapped to either 0 or AABB_MAX_FVALUE so that we can identify them in shiftAllObjects.
	// We add 1 to the AABB max the same as in convertAabbToBroadPhaseResolution.
	if (border)
	{
		for (int i = 0; i < 3; i++)
		{
			aabb.m_min[i] = (aabb.m_min[i] < AABB_MAX_FVALUE / 2) ? 0 : AABB_MAX_FVALUE;
			aabb.m_max[i] = 1 + ((aabb.m_max[i] < AABB_MAX_FVALUE / 2) ? 0 : AABB_MAX_FVALUE);
		}
	}

	// add the object to the end of the list
	const int nodeIndex = m_nodes.getSize();

	hkpBpNode& node = m_nodes.expandOne();

	// insert into all axis
	hkpBpNode* nodes = &m_nodes[0];
	m_axis[0].insert( nodes, nodeIndex, aabb.m_min[0], aabb.m_max[0], node.min_x, node.max_x  );
	m_axis[1].insert( nodes, nodeIndex, aabb.m_min[1], aabb.m_max[1], node.min_y, node.max_y  );
	m_axis[2].insert( nodes, nodeIndex, aabb.m_min[2], aabb.m_max[2], node.min_z, node.max_z  );

	updateNodesAfterInsert( nodes, nodeIndex, node );

	node.m_handle = object;
	object->m_id = nodeIndex;


	//
	// now find all overlapping objects
	//
	{
		// clear the scratch area
		int numNodes = m_nodes.getSize();
		hkArray<hkUint32>::Temp scratchArea( (numNodes>>5)+8 );
		hkUint32 *bitField = scratchArea.begin();

		setBitsBasedOnXInterval( numNodes, aabb.m_min[0], node, nodeIndex, bitField);
		querySingleAabbAddObject( object, nodeIndex, bitField, node, newPairs );
	}

	CHECK_CONSISTENCY();

}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::addObject( hkpBroadPhaseHandle* object, const hkAabb& aabbIn, hkArray<hkpBroadPhaseHandlePair>& newPairs, bool border )
{
	HK_ALIGN16( hkAabbUint32 aabb32 );
	// We need to convert the AABB to a regular 32bit integer AABB as the other addObject() function will perform a conversion from 32bit to the broadphase's resolution.
	hkAabbUtil::convertAabbToUint32( aabbIn, m_offsetLow32bit, m_offsetHigh32bit, m_scale32bit, aabb32 );
	addObject( object, aabb32, newPairs, border );
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::querySingleAabbRemoveObject( hkpBroadPhaseHandle* object, int newNodeIndex, const hkUint32 *bitField, hkpBpNode& refNode, hkArray<hkpBroadPhaseHandlePair>& pairs_out) const
{
	hkpBroadPhaseHandlePair pair;
	pair.m_a = object;

	const hkpBpNode *node = &m_nodes[0];
	const hkUint32* end = bitField + (m_nodes.getSize() >> 5) + 1;

	for( ; bitField < end; node += 32, bitField++ ) //check 32 at a time
	{
		const hkpBpNode *n = node;
		for ( hkUint32 mask = bitField[0]; mask; )
		{
			if ( 0 == (mask & 0xff) )
			{
				n += 8;
				mask = mask>>8;
				continue;
			}
			if ( ( mask & 0x1 ) && !n->yzDisjoint( refNode ) )
			{
				if ( !n->isMarker() )
				{
					pair.m_b = n->m_handle;
					pairs_out.pushBack( pair );
				}
				else
				{
					hkpBpMarker& m = n->getMarker( m_markers );
					int i = m.m_overlappingObjects.indexOf( newNodeIndex );
					m.m_overlappingObjects.removeAt(i);
				}
			}
			n += 1;
			mask = mask>>1;
		}
	}

	//
	//	Do some cross checking
	//
//	if(0)
//	{
//		for (int i = 0 ;i < m_nodes.getSize(); i++)
//		{
//			if ( i == newNodeIndex)
//			{
//				continue;
//			}
//			const hkpBpNode& n = m_nodes[i];
//			if ( !n.isMarker() )
//			{
//				hkBool disjoint = refNode.xyDisjoint(n) || refNode.yzDisjoint(n);
//				if (!disjoint)
//				{
//					int j;
//					for ( j = 0; j < pairs_out.getSize(); j++)
//					{
//						if ( pairs_out[j].m_b == n.m_handle )
//						{
//							break;
//						}
//					}
//					HK_ASSERT(0x1bb6c143, j < pairs_out.getSize());
//				}
//			}
//		}
//	}
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::removeObject(   hkpBroadPhaseHandle* object, hkArray<hkpBroadPhaseHandlePair>& delPairsOut )
{
	//CHECK_CONSISTENCY();
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

		// add the object to the end of the list
	hkUint32 numNodes = m_nodes.getSize();
	const hkUint32 nodeIndex = object->m_id;
	HK_ASSERT2(0xad674dbb, nodeIndex > 0, "Invalid bp handle as parameter.");
	hkpBpNode& node = m_nodes[nodeIndex];		// the node to delete
	//
	//	find overlapping objects
	//
	{
		// clear the scratch area
		hkArray<hkUint32>::Temp scratchArea( (numNodes>>5)+8 );

		hkUint32 *bitField = scratchArea.begin();
		setBitsBasedOnXInterval( numNodes, m_axis[0].m_endPoints[node.min_x].m_value, node, nodeIndex, bitField);
		querySingleAabbRemoveObject( object, nodeIndex, bitField, node, delPairsOut);
	}


	// remove from all axis
	hkpBpNode* nodes = &m_nodes[0];
	m_axis[0].remove( node.min_x, node.max_x  );
	m_axis[1].remove( node.min_y, node.max_y  );
	m_axis[2].remove( node.min_z, node.max_z  );
	updateNodesAfterDelete( nodes, numNodes, node );

	//
	//	move the last node in the nodes array to the position of the deleted node
	//  and relink
	//
	if ( nodeIndex < numNodes-1)
	{
		node = m_nodes[ numNodes-1];
		m_axis[0].m_endPoints[ node.min_x ].m_nodeIndex = nodeIndex;
		m_axis[0].m_endPoints[ node.max_x ].m_nodeIndex = nodeIndex;
		if ( !node.isMarker() )
		{
			m_axis[1].m_endPoints[ node.min_y ].m_nodeIndex = nodeIndex;
			m_axis[1].m_endPoints[ node.max_y ].m_nodeIndex = nodeIndex;
			m_axis[2].m_endPoints[ node.min_z ].m_nodeIndex = nodeIndex;
			m_axis[2].m_endPoints[ node.max_z ].m_nodeIndex = nodeIndex;
			node.m_handle->m_id = nodeIndex;
		}
		else
		{
			// we do not update the nodeIndex of the endPoints, as they point to node 0
			HK_ASSERT(0x5cfb9710,  0 == m_axis[1].m_endPoints[ node.min_y ].m_nodeIndex );
			HK_ASSERT(0x622bf10c,  0 == m_axis[1].m_endPoints[ node.max_y ].m_nodeIndex );
			HK_ASSERT(0x4c39a0cc,  0 == m_axis[2].m_endPoints[ node.min_z ].m_nodeIndex );
			HK_ASSERT(0x6ed915a4,  0 == m_axis[2].m_endPoints[ node.max_z ].m_nodeIndex );
			node.getMarker( m_markers).m_nodeIndex = nodeIndex;
		}


		//
		//	Update the overlap lists in the markers of the moved node
		//
		if (getNumMarkers() && !node.isMarker() )
		{
			// find marker
			int minPos = (m_axis[0].m_endPoints[node.min_x].m_value>>(HK_BP_NUM_VALUE_BITS-m_ld2NumMarkers));
			if ( minPos > 0)
			{
				hkpBpNode& markerNode = m_nodes[ m_markers[minPos-1].m_nodeIndex ];
				if ( markerNode.max_x > node.min_x )
				{
					minPos--;
				}
			}

			int maxPos = (m_axis[0].m_endPoints[node.max_x].m_value>>(HK_BP_NUM_VALUE_BITS-m_ld2NumMarkers))-1;
			if ( maxPos >= 0)
			{
				HK_ON_DEBUG(hkpBpNode& markerNode0 = m_nodes[ m_markers[maxPos].m_nodeIndex ]);
				HK_ASSERT(0x573f12ba,  markerNode0.min_x < node.max_x  );
				if ( maxPos < m_numMarkers-1)
				{
					HK_ON_DEBUG(hkpBpNode& markerNode1 = m_nodes[ m_markers[maxPos+1].m_nodeIndex ]);
					HK_ASSERT(0x6cd04f19,  markerNode1.min_x > node.max_x  );
				}
			}
			for ( int m = minPos; m <= maxPos; m++)
			{
				hkpBpMarker& marker = m_markers[m];
				int i = marker.m_overlappingObjects.indexOf(numNodes-1);
				marker.m_overlappingObjects[i] = nodeIndex;
			}
		}
#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
		_fixDeterministicOrderAfterNodeIdWasDecreased( nodeIndex );
#endif
	}
	m_nodes.setSize( numNodes-1 );

	CHECK_CONSISTENCY();
}
#endif

	/// returns the bp index of start of the tail
int hkp3AxisSweep::hkpBpAxis::insertTail( hkpBpNode *nodes, int axis, hkpBpEndPoint* newNodes, int numNewEndPoints )
{
	int oldSize = m_endPoints.getSize();
	int newSize = oldSize + numNewEndPoints;
	m_endPoints.setSize( newSize );

	BpInt maxVal = newNodes[numNewEndPoints-1].m_value;

	// copy the old end, do not update the node
	hkpBpEndPoint* dest   = m_endPoints.begin() + newSize -1;
	hkpBpEndPoint* source = m_endPoints.begin() + oldSize -1;

	hkMath::prefetch128( dest );
	hkMath::prefetch128( dest-128 );
	hkMath::prefetch128( source );
	hkMath::prefetch128( dest-128 );
	{
		*dest = *source;	// copy last element
		nodes[ source->m_nodeIndex ].setElem( axis, source->isMaxPoint(), dest - &m_endPoints[0] );
		source--;
		dest--;
	}

	while ( source->m_value > maxVal )
	{
		*dest = *source;	// copy rest
		hkMath::prefetch128( source - 128 );

		nodes[ source->m_nodeIndex ].setElem( axis, source->isMaxPoint(), dest - &m_endPoints[0] );

		hkMath::prefetch128( source - 128 );
		source--;
		dest--;

	}
	return (dest - m_endPoints.begin()) + 1;
}


template< int axis >
void hkp3AxisSweep::hkpBpAxis::mergeRest( hkpBpNode *nodes, int startOfTail, hkpBpEndPoint* newEndPoints, int numNewEndPoints )
{
	const hkpBpEndPoint *sourceNew = newEndPoints + numNewEndPoints -1;
	const hkpBpEndPoint *sourceOld = m_endPoints.begin() + (startOfTail - numNewEndPoints - 1 );
	int destIndex = startOfTail - 1;
	hkpBpEndPoint *dest = m_endPoints.begin() + destIndex;


	int oldVal = sourceOld->m_value;
	int newVal = sourceNew->m_value;
	while(1)
	{
		if( oldVal > newVal )
		{
			*dest = *sourceOld;		// this case is more likely
			nodes[ sourceOld->m_nodeIndex ].setElem( axis, sourceOld->isMaxPoint(), destIndex );
			sourceOld--;	// no need to check for underflow, as there oldVal always contains the smallest element 0
			dest--;
			destIndex--;
			oldVal = sourceOld->m_value;	
		}
		else
		{
			*dest = *sourceNew;
			nodes[ sourceNew->m_nodeIndex ].setElem( axis, sourceNew->isMaxPoint(), destIndex );
			sourceNew--;
			dest--;
			destIndex--;

			if ( sourceNew <  newEndPoints )
			{
				break;
			}

			
			newVal = sourceNew->m_value;
		}
	}
	// the rest should be correct
}

#if !defined(HK_PLATFORM_SPU)




void hkp3AxisSweep::addObjectBatch( const hkArrayBase<hkpBroadPhaseHandle*>& addObjectList, const hkArrayBase<hkAabb>& addAabbList, hkArray<hkpBroadPhaseHandlePair>& newPairs )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	if( addAabbList.getSize() < 1 )
	{
		return;
	}

	HK_INTERNAL_TIMER_BEGIN_LIST( "AddBatch", "init");
	const int oldNumNodes = m_nodes.getSize();
	const int numNewNodes = addObjectList.getSize();
	const int numNewEndPoints = numNewNodes + numNewNodes;

	hkArray<hkpBpEndPoint>::Temp endPointsX(numNewEndPoints);
	hkArray<hkpBpEndPoint>::Temp endPointsY(numNewEndPoints);
	hkArray<hkpBpEndPoint>::Temp endPointsZ(numNewEndPoints);
	hkArray<const hkpBpNode*>::Temp inAabbNodes;
	inAabbNodes.reserveExactly( oldNumNodes );

	// resize arrays
	m_nodes.expandBy( numNewNodes );
	hkAabb combinedAabb; combinedAabb.setEmpty();
	{
		hkpBpEndPoint *newEps[3];
		newEps[0] = endPointsX.begin();
		newEps[1] = endPointsY.begin();
		newEps[2] = endPointsZ.begin();

		// copy data to new array spaces
		for( int nodeIndex = 0; nodeIndex < numNewNodes; nodeIndex++ )
		{
			combinedAabb.includeAabb(addAabbList[nodeIndex]);
			HK_ALIGN16( hkAabbUint32 aabb );		// the int version
			_convertAabbToInt( addAabbList[nodeIndex], aabb );

			int nodeIdx = oldNumNodes + nodeIndex;
			hkpBpNode& node = m_nodes[ nodeIdx ];
			hkpBroadPhaseHandle *object = addObjectList[nodeIndex];
			node.m_handle = object;
			object->m_id = nodeIdx;

			endPointsX[nodeIndex+nodeIndex].m_value = aabb.m_min[0];
			endPointsX[nodeIndex+nodeIndex].m_nodeIndex = nodeIdx;
			endPointsX[nodeIndex+nodeIndex+1].m_value = aabb.m_max[0];
			endPointsX[nodeIndex+nodeIndex+1].m_nodeIndex = nodeIdx;

			endPointsY[nodeIndex+nodeIndex].m_value = aabb.m_min[1];
			endPointsY[nodeIndex+nodeIndex].m_nodeIndex = nodeIdx;
			endPointsY[nodeIndex+nodeIndex+1].m_value = aabb.m_max[1];
			endPointsY[nodeIndex+nodeIndex+1].m_nodeIndex = nodeIdx;

			endPointsZ[nodeIndex+nodeIndex].m_value = aabb.m_min[2];
			endPointsZ[nodeIndex+nodeIndex].m_nodeIndex = nodeIdx;
			endPointsZ[nodeIndex+nodeIndex+1].m_value = aabb.m_max[2];
			endPointsZ[nodeIndex+nodeIndex+1].m_nodeIndex = nodeIdx;
		}
		HK_INTERNAL_TIMER_SPLIT_LIST("scanAabb");

		// remember all old nodes which are in the combined AABB
		{
			m_nodes.setSizeUnchecked(oldNumNodes);
			_querySingleAabb( combinedAabb, HK_BP_REPORT_NODES, HK_NULL, &inAabbNodes, HK_NULL);
			m_nodes.expandBy( numNewNodes );
		}

		HK_INTERNAL_TIMER_SPLIT_LIST("sort");

		// sort the new points, than merge
		hkpBpNode* nodes = &m_nodes[0];
		{
			int startOfTail[3];
			{
				for (int a = 0; a < 3; a++)
				{
					hkSort( newEps[a], numNewEndPoints );
				}
			}
			{
				HK_INTERNAL_TIMER_SPLIT_LIST("insertTail");
				for (int a = 0; a < 3; a++)
				{
					startOfTail[a] = m_axis[a].insertTail( nodes, a, newEps[a], numNewEndPoints );
				}
			}


// 			HK_INTERNAL_TIMER_SPLIT_LIST("updateNodesAfterTailInsert");
// 			int startOfTailOld[3] = { startOfTail[0]-numNewEndPoints, startOfTail[1]-numNewEndPoints, startOfTail[2]-numNewEndPoints };
// 			updateNodesAfterBatchTailInsert( nodes, oldNumNodes, numNewEndPoints, startOfTailOld );	// already done in insertTail()


			HK_INTERNAL_TIMER_SPLIT_LIST("merge");
			m_axis[0].mergeRest<0>( nodes, startOfTail[0], newEps[0], numNewEndPoints );
			m_axis[1].mergeRest<1>( nodes, startOfTail[1], newEps[1], numNewEndPoints );
			m_axis[2].mergeRest<2>( nodes, startOfTail[2], newEps[2], numNewEndPoints );
		}

		//
		//	Update the markers for axis yz
		//   Note: the markers span the entire broadphase for the y/z axis, the x-axis is paper thin
		//
		if (getNumMarkers())
		{
			HK_INTERNAL_TIMER_SPLIT_LIST("markers");

			//int numNewEndPoints = numNew * 2;
			for (int im = 0; im < getNumMarkers(); im++)
			{
				hkpBpMarker& m = m_markers[im];
				hkpBpNode&n = m_nodes[m.m_nodeIndex];
				n.max_y += numNewEndPoints;
				n.max_z += numNewEndPoints;
			}
		}
	}
	CHECK_CONSISTENCY();


	// find new overlaps
	if (0)
	{
		HK_INTERNAL_TIMER_SPLIT_LIST("findOverlapsOld");

		int numBits = m_nodes.getSize();
		int numBytes =     (numBits>>3);
		HK_ASSERT2( 0x234feba2, false, "Change the allocation below to a Temp allocation" );
		int numInts  = 8 + (numBits>>5);
		hkUint32* bitFieldOfQueryNodes = hkAllocateStack<hkUint32>( numInts );
		memclear16( bitFieldOfQueryNodes, numBytes );

		for( int i = 0; i < numNewNodes; i++ )
		{
			staticFlipBit( bitFieldOfQueryNodes, oldNumNodes + i);
		}

		queryBatchAabbSub( bitFieldOfQueryNodes, newPairs, true );
		hkDeallocateStack(bitFieldOfQueryNodes, numInts);
	}
#if defined(HK_BROADPHASE_32BIT)
#	define SORT_TYPE SortData32
#	define SORT_FUNCTION sort32
#else
#	define SORT_TYPE SortData16
#	define SORT_FUNCTION sort16
#endif

	// new version
	{
		HK_INTERNAL_TIMER_SPLIT_LIST("sortNewNodes");

		// create a sorted new node array
		hkArray<hkpBpNode>::Temp sortedNewNodes( numNewNodes + 1);
		{
			hkArray<hkRadixSort::SORT_TYPE>::Temp sortData( numNewNodes + 4 );
			hkArray<hkRadixSort::SORT_TYPE>::Temp sortBuffer( numNewNodes + 4 );
			{for (int i = 0; i < numNewNodes; i++){ sortData[i].m_userData = oldNumNodes + i; sortData[i].m_key = m_nodes[oldNumNodes + i].min_x; }}
			sortData[numNewNodes].m_key = BpInt(-1);sortData[numNewNodes+1].m_key = BpInt(-1);sortData[numNewNodes+2].m_key = BpInt(-1);
			hkRadixSort::SORT_FUNCTION( sortData.begin(), HK_NEXT_MULTIPLE_OF(4,numNewNodes), sortBuffer.begin() );
			{for (int i = 0; i < numNewNodes; i++){ sortedNewNodes[ i ] = m_nodes[ sortData[i].m_userData ]; }}

			hkpBpNode& lastDummy = sortedNewNodes[numNewNodes];
			lastDummy.min_x = BpInt(-1);
		}

		// extract a list of sorted old + new  nodes
		int numRefNodes = inAabbNodes.getSize(); 
		hkArray<hkpBpNode>::Temp sortedRefNodes( numRefNodes + 1);

		HK_INTERNAL_TIMER_SPLIT_LIST("sortRefNodes");
		{
			hkArray<hkRadixSort::SORT_TYPE>::Temp sortData( numRefNodes + 4 );
			hkArray<hkRadixSort::SORT_TYPE>::Temp sortBuffer( numRefNodes + 4 );

			hkpBpNode* nodes = m_nodes.begin();
			{for (int i = 0; i < numRefNodes; i++){ const hkpBpNode* node = inAabbNodes[i]; sortData[i].m_userData = node-nodes; sortData[i].m_key = node->min_x; }}
			sortData[numRefNodes].m_key = BpInt(-1);sortData[numRefNodes+1].m_key = BpInt(-1);sortData[numRefNodes+2].m_key = BpInt(-1);
			hkRadixSort::SORT_FUNCTION( sortData.begin(), HK_NEXT_MULTIPLE_OF(4,numRefNodes), sortBuffer.begin() );
			{for (int i = 0; i < numRefNodes; i++){ sortedRefNodes[ i ] = m_nodes[ sortData[i].m_userData ]; }}

			hkpBpNode& lastDummy = sortedRefNodes[numRefNodes];
			lastDummy.min_x = BpInt(-1);
		}
		HK_INTERNAL_TIMER_SPLIT_LIST("selfCollide");

		// collide against itself
		collide1Axis( sortedNewNodes.begin(), numNewNodes, newPairs );

		HK_INTERNAL_TIMER_SPLIT_LIST("otherCollide");

		// collide with existing objects
		collide1Axis( sortedNewNodes.begin(), numNewNodes, sortedRefNodes.begin(), numRefNodes, MARKERS_ADD_NEW_OVERLAPS, newPairs );
	}
	HK_INTERNAL_TIMER_END_LIST();

}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::removeObjectBatch( const hkArrayBase<hkpBroadPhaseHandle*>& removeObjectList, hkArray<hkpBroadPhaseHandlePair>& delPairsOut )
{
	HK_ASSERT2( 0xf0dfed23, removeObjectList.getSize(), "You cannot call hkp3AxisSweep::removeObjectBatch() with 0 objects" );
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	HK_INTERNAL_TIMER_BEGIN_LIST("removeObjectBatch", "init");
	int numRemoveNodes = removeObjectList.getSize();

	// do a batch query
	int numNodes = m_nodes.getSize();

	// find old overlaps
	if (0)
	{
		int numBits = numNodes;
		int numBytes =     (numBits>>3);
		int numInts  = 8 + (numBits>>5);
		HK_ASSERT2( 0x234feba2, false, "Change the allocation below to a Temp allocation" );
		hkUint32* bitFieldOfQueryNodes = hkAllocateStack<hkUint32>( numInts );
		memclear16( bitFieldOfQueryNodes, numBytes );

		for( int i = 0; i < numRemoveNodes; i++ )
		{
			hkpBroadPhaseHandle *object = removeObjectList[i];
			staticFlipBit( bitFieldOfQueryNodes, object->m_id);
		}
		queryBatchAabbSub( bitFieldOfQueryNodes, delPairsOut, false );

		hkDeallocateStack(bitFieldOfQueryNodes, numInts);
	}


	HK_INTERNAL_TIMER_SPLIT_LIST("killList");

		// now remove objects from the data structures
		// create a sorted list of those copied nodes
		// and build the max AABB
	hkpBroadPhaseHandle notExistingHandle;
	hkArray<hkpBpNode>::Temp sortedRemoveNodes( numRemoveNodes + 1);
	hkpBpNode maxExtents; maxExtents = m_nodes[removeObjectList[0]->m_id];	// the max aabb
	{
		hkArray<hkRadixSort::SORT_TYPE>::Temp sortData( numRemoveNodes + 4 );
		{
			for( int i = 0; i < numRemoveNodes; i++ )
			{
				const int nodeIndex = removeObjectList[i]->m_id;
				hkpBpNode& node = m_nodes[nodeIndex];		// the node to delete
				sortData[i].m_userData = nodeIndex;
				sortData[i].m_key = node.min_x;
				maxExtents.min_x = hkMath::min2( maxExtents.min_x, node.min_x );
				maxExtents.min_y = hkMath::min2( maxExtents.min_y, node.min_y );
				maxExtents.min_z = hkMath::min2( maxExtents.min_z, node.min_z );
				maxExtents.max_x = hkMath::max2( maxExtents.max_x, node.max_x );
				maxExtents.max_y = hkMath::max2( maxExtents.max_y, node.max_y );
				maxExtents.max_z = hkMath::max2( maxExtents.max_z, node.max_z );
			}
		}

		{
			hkArray<hkRadixSort::SORT_TYPE>::Temp sortBuffer( numRemoveNodes + 4 );
			sortData[numRemoveNodes].m_key = BpInt(-1);sortData[numRemoveNodes+1].m_key = BpInt(-1);sortData[numRemoveNodes+2].m_key = BpInt(-1);
			hkRadixSort::SORT_FUNCTION( sortData.begin(), HK_NEXT_MULTIPLE_OF(4,numRemoveNodes), sortBuffer.begin() );
			{
				for (int i = 0; i < numRemoveNodes; i++)
				{
					hkpBpNode& node = m_nodes[sortData[i].m_userData];		// the node to delete
					sortedRemoveNodes[ i ] = node;

					// mark node for deletion
					node.m_handle->m_id = HK_NULL;
					node.m_handle = &notExistingHandle;
				}
			}

			hkpBpNode& lastDummy = sortedRemoveNodes[numRemoveNodes];
			lastDummy.min_x = BpInt(-1);
		}
	}
	HK_INTERNAL_TIMER_SPLIT_LIST("createMapping");

		// kill nodes. create mapping from old node positions to new ones
		// we use the value of -1 if the node is deleted
#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
	// an array of nodes which have been moved, need for sorting to keep deterministic order
	hkArray<int>::Temp movedNodes;
	movedNodes.reserveExactly( numRemoveNodes + getNumMarkers() );	
#endif
	hkArray<int>::Temp nodeRelocations( numNodes );

	int numRefNodes = 0; 
	hkArray<hkRadixSort::SORT_TYPE>::Temp sortData( numNodes - numRemoveNodes + 4 );	// the indices to a node which overlaps with the maxExtents
	{
			// for all deleted nodes replace it by a node at the end of the array
		int endElement = numNodes - 1;
		nodeRelocations[0] = 0;	// skip the very first node as this can never collide with anything
		for( int i = 1; i <= endElement;  i++)
		{
				// check for deleted nodes at the end of the array, we need to do this, as we cannot move moved elements
			if ( m_nodes[endElement].m_handle == &notExistingHandle)
			{
				nodeRelocations[ endElement ] = -1;
				endElement--;
				i--;
				continue;
			}

				// the node to delete, copy end node to this place
			hkpBpNode& node = m_nodes[i];
			if ( node.m_handle == &notExistingHandle)
			{
				node = m_nodes[ endElement ];	// copy values

				nodeRelocations[endElement] = i;
				nodeRelocations[i] = -1;
				endElement--;
#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
				movedNodes.pushBackUnchecked( i );
#endif

				if ( !node.isMarker())
				{
					node.m_handle->m_id = i;
				}
				else
				{
					hkpBpMarker& m = node.getMarker( m_markers );
					m.m_nodeIndex = i;
					continue;
				}
			}
			else
			{
				nodeRelocations[i] = i;
			}
			if ( !maxExtents.yzDisjoint(node) && !maxExtents.xyDisjoint(node) )
			{
				sortData[numRefNodes].m_key = node.min_x;
				sortData[numRefNodes++].m_userData = i;
			}
		}
		numNodes = endElement + 1;
		HK_ASSERT(0x592e88cf,  numRemoveNodes == m_nodes.getSize() - numNodes);
		m_nodes.setSize( numNodes );
	}

	hkArray<hkpBpNode>::Temp sortedRefNodes( numRefNodes + 1);

	HK_INTERNAL_TIMER_SPLIT_LIST("sortRefNodes");
	{
		hkArray<hkRadixSort::SORT_TYPE>::Temp sortBuffer( numRefNodes + 4 );
		sortData[numRefNodes].m_key = BpInt(-1);sortData[numRefNodes+1].m_key = BpInt(-1);sortData[numRefNodes+2].m_key = BpInt(-1);
		hkRadixSort::SORT_FUNCTION( sortData.begin(), HK_NEXT_MULTIPLE_OF(4,numRefNodes), sortBuffer.begin() );
		{for (int i = 0; i < numRefNodes; i++){ sortedRefNodes[ i ] = m_nodes[ sortData[i].m_userData ]; }}

		hkpBpNode& lastDummy = sortedRefNodes[numRefNodes];
		lastDummy.min_x = BpInt(-1);
	}

	HK_INTERNAL_TIMER_SPLIT_LIST("selfCollide");

	{
		//hkArray<hkpBroadPhaseHandlePair> delPairsOutTest;
		// collide against itself
		collide1Axis( sortedRemoveNodes.begin(), numRemoveNodes, delPairsOut );

		HK_INTERNAL_TIMER_SPLIT_LIST("otherCollide");

		// collide with existing objects
		collide1Axis( sortedRemoveNodes.begin(), numRemoveNodes, sortedRefNodes.begin(), numRefNodes, MARKERS_IGNORE, delPairsOut );
	}


	HK_INTERNAL_TIMER_SPLIT_LIST("fixupAxis");

		// remove from this axis
	m_axis[0].removeBatch<0>( &m_nodes[0], nodeRelocations );
	m_axis[1].removeBatch<1>( &m_nodes[0], nodeRelocations );
	m_axis[2].removeBatch<2>( &m_nodes[0], nodeRelocations );


	//
	//	Update markers for axis y-z and marker pointers
	//
	if (getNumMarkers())
	{
		HK_INTERNAL_TIMER_SPLIT_LIST("Markers");
		int numRemoveEndPoints = numRemoveNodes * 2;
		for (int im = 0; im < getNumMarkers(); im++)
		{
			hkpBpMarker& m = m_markers[im];
			hkpBpNode&n = m_nodes[m.m_nodeIndex];
			n.max_y -= numRemoveEndPoints;
			n.max_z -= numRemoveEndPoints;

			int dest = 0;
			for (int i = 0; i < m.m_overlappingObjects.getSize(); i++)
			{
				int newIndex = nodeRelocations[ m.m_overlappingObjects[i]];
				if ( newIndex >= 0 )
				{
					m.m_overlappingObjects[dest++] = newIndex;
				}
			}
			m.m_overlappingObjects.setSize(dest);
		}
	}

	HK_INTERNAL_TIMER_END_LIST();

#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
	{
		for (int i =0 ; i < movedNodes.getSize(); i++ )
		{
			_fixDeterministicOrderAfterNodeIdWasDecreased( movedNodes[i] );
		}
	}
#endif
	//HK_INTERNAL_TIMER_END_LIST();

	CHECK_CONSISTENCY();

}
#endif


#if !defined(HK_PLATFORM_SPU)
int hkp3AxisSweep::getNumObjects() const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	return m_nodes.getSize() - 1;
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::getAllAabbs( hkArray<hkAabb>& allAabbs ) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );

	allAabbs.reserveExactly( m_nodes.getSize()   - m_numMarkers );
	allAabbs.setSizeUnchecked( m_nodes.getSize() - m_numMarkers );

	// Loop and grab
	int d = 0;
	for ( int i=0; i < m_nodes.getSize();i++)
	{
		const hkpBpNode &node = m_nodes[i];
		if ( !node.isMarker() )
		{
			getAabbFromNode(node,allAabbs[d++]);
		}
	}
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::getAabb(const hkpBroadPhaseHandle* object, hkAabb& aabb) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );

	const hkpBpNode* nodes = &m_nodes[0];
	int nodeIndex = hkUlong(object->m_id);
	const hkpBpNode& node = nodes[ nodeIndex ];
	getAabbFromNode(node,aabb);
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::getExtents(hkVector4& worldMinOut, hkVector4& worldMaxOut) const
{
	HK_ASSERT2(0xaf542fe2, m_scale.length<4>().getReal() != 0, "Make sure to call set32BitOffsetAndScale() after creating the broadphase.");

    worldMinOut.setNeg<4>(m_offsetLow);
	worldMaxOut.setReciprocal(m_scale); 
	worldMaxOut.setComponent<3>(hkSimdReal_1);
	worldMaxOut.mul(hkSimdReal::fromFloat(hkReal(AABB_MAX_FVALUE)));
    worldMaxOut.add(worldMinOut);
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::checkConsistency()
{
#ifdef HK_DEBUG
	{
		if ( m_nodes.getCapacity()> m_nodes.getSize())
		{
			m_nodes.end()->min_x = BpInt(-1);
			m_nodes.end()->max_x = BpInt(-1);
			m_nodes.end()->min_y = BpInt(-1);
			m_nodes.end()->max_y = BpInt(-1);
			m_nodes.end()->min_z = BpInt(-1);
			m_nodes.end()->max_z = BpInt(-1);
			m_nodes.end()->m_handle = HK_NULL;
		}
		for ( int a = 0; a < 3; a++)
		{
			if ( m_axis[a].m_endPoints.getCapacity() > m_axis[a].m_endPoints.getSize() )
			{
				m_axis[a].m_endPoints.end()->m_nodeIndex = BpInt(-1);
				m_axis[a].m_endPoints.end()->m_value = 0;
			}
		}
	}

	// check the order of the endpoints
	HK_ON_DEBUG( hkpBpNode* nodes = m_nodes.begin() );
	for ( int i=0; i < m_nodes.getSize(); i++)
	{
		hkpBpNode &node = m_nodes[i];
		// check all axis
		for (int axisIndex = 0; axisIndex < 3; axisIndex++)
		{
			hkpBpAxis &axis = m_axis[axisIndex];
			hkArray<hkpBpEndPoint>& eps = axis.m_endPoints;
			{ // check max
				int epIndex = node.getMax(axisIndex);
				hkpBpEndPoint* ep = &eps[ epIndex ];
				HK_ASSERT(0x75d33e02,  ep[0].isMaxPoint() == 1 );
				if ( !node.isMarker() || axisIndex==0 )
				{
					HK_ASSERT(0x3054309b,  (nodes + ep[0].m_nodeIndex) == &node );
					if ( i ) // not special guard node
					{
#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
						HK_ASSERT(0x10627724,  ep[-1].m_value < ep[0].m_value || ( ep[-1].m_value == ep[0].m_value && ep[-1].m_nodeIndex < ep[0].m_nodeIndex));
						HK_ASSERT(0x3e9a80b1,  ep[ 0].m_value < ep[1].m_value || ( ep[ 0].m_value == ep[1].m_value && (ep[ 0].m_nodeIndex < ep[1].m_nodeIndex || ep[1].m_nodeIndex == 0)));
#else
						HK_ASSERT(0x10627724,  ep[-1].m_value <= ep[0].m_value );
						HK_ASSERT(0x3e9a80b1,  ep[ 0].m_value <= ep[1].m_value );
#endif
					}
					else
					{
						HK_ASSERT(0x619b2a33,  epIndex == axis.m_endPoints.getSize()-1 );	// check marker node
					}
				}
				else
				{
					HK_ASSERT(0x619b2a33,  epIndex == axis.m_endPoints.getSize()-1 );	// check marker node
				}
			}
			{ // check min
				int epIndex = node.getMin(axisIndex);
				hkpBpEndPoint* ep = &eps[ epIndex ];
				HK_ASSERT(0x4d2b1d03,  ep[0].isMaxPoint() == 0 );
				if ( !node.isMarker() || axisIndex==0 )
				{
					HK_ON_DEBUG( hkpBpEndPoint *e = &axis.m_endPoints[ epIndex ] );
					HK_ASSERT(0x6f5b9c43,  (nodes + e[0].m_nodeIndex) == &node );
					if ( i ) // not special node
					{
#if HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED
						HK_ASSERT(0x7313f96f,  e[-1].m_value < e[0].m_value || ( ep[-1].m_value == ep[0].m_value && ep[-1].m_nodeIndex < ep[0].m_nodeIndex));
						HK_ASSERT(0x7dca990c,  e[ 0].m_value < e[1].m_value || ( ep[ 0].m_value == ep[1].m_value && ep[ 0].m_nodeIndex < ep[1].m_nodeIndex));
#else
						HK_ASSERT(0x7313f96f,  e[-1].m_value < e[0].m_value );
						HK_ASSERT(0x7dca990c,  e[ 0].m_value < e[1].m_value );
#endif					
					}
					else
					{
						HK_ASSERT(0x619b2a33,  epIndex == 0 );	// check marker node
					}
				}
				else
				{
					HK_ASSERT(0x70b73de4,  epIndex == 0 );
				}
			}
			if ( axisIndex == 0)
			{
				HK_ASSERT(0x35d28b6b,  axis.m_endPoints.getSize() == m_nodes.getSize() * 2 );
			}
			else
			{
				HK_ASSERT(0x29a1f322,  axis.m_endPoints.getSize() == ( m_nodes.getSize()- getNumMarkers()) * 2 );
			}
		}
	}

	// check the reverse mappings
	{
		for ( int j=1; j<m_nodes.getSize();j++)
		{
			hkpBpNode &node = m_nodes[j];
			if ( !node.isMarker())
			{
				HK_ASSERT(0x332f0c19,  node.m_handle->m_id == unsigned(j) );
			}
			else
			{
				HK_ASSERT(0x21503b41, int(node.getMarker(m_markers).m_nodeIndex) == j);
			}
		}
	}

#if defined(HK_DEBUG)
	//
	//	Check the marker consistency (pretty expensive)
	//
	if(1)
	{
		for (int im = 0; im < getNumMarkers(); im++)
		{
			hkpBpMarker& m = m_markers[im];
			hkpBpNode& mn = m_nodes[m.m_nodeIndex];

			for ( int in = 1; in < m_nodes.getSize(); in++)
			{
				hkpBpNode& n = m_nodes[in];
				if ( !n.isMarker() )
				{
					hkBool disjoint = mn.xyDisjoint(n) || mn.yzDisjoint(n);
					hkBool isInMarker = m.m_overlappingObjects.indexOf(in) >=0;
					HK_ASSERT(0x549bd45a,  disjoint != isInMarker);
				}
			}
		}
	}
#endif

	// check the first node
	HK_ASSERT(0x694a00a9,  nodes->min_x == 0 );
	HK_ASSERT(0x41b3a5bb,  nodes->min_y == 0 );
	HK_ASSERT(0x7277fbb3,  nodes->min_z == 0 );
	HK_ASSERT(0x2879fcd2,  int(nodes->max_x) == m_nodes.getSize()*2-1 );
	HK_ASSERT(0x14fe2bf6,  int(nodes->max_y) == (m_nodes.getSize()-m_numMarkers)*2-1 );
	HK_ASSERT(0x598f3c74,  int(nodes->max_z) == (m_nodes.getSize()-m_numMarkers)*2-1 );
#endif
}
#endif

#ifndef HK_PLATFORM_SPU
static HK_ALWAYS_INLINE hkp3AxisSweep::BpInt getEpValue(const hkp3AxisSweep::hkpBpEndPoint* ep)
{
	return ep->m_value;
}
#else
static HK_ALWAYS_INLINE hkp3AxisSweep::BpInt getEpValue(const hkp3AxisSweep::hkpBpEndPoint* ep)
{
	extern hkSpu4WayCache* g_SpuCollideUntypedCache;
	const int dmaGroup = HK_SPU_DMA_GROUP_STALL;
	const hkBool waitForDmaCompletion = true;

	const hkp3AxisSweep::hkpBpEndPoint* spuEp = hkGetArrayElemUsingCache(ep, 0, g_SpuCollideUntypedCache, HK_SPU_UNTYPED_CACHE_LINE_SIZE, dmaGroup, waitForDmaCompletion);
	return spuEp->m_value;
}
#endif
const hkp3AxisSweep::hkpBpEndPoint* hkp3AxisSweep::hkpBpAxis::find( const hkpBpEndPoint* start, const hkpBpEndPoint* end, BpInt value) const
{
	while ( end - start > 16 )
	{
		const hkpBpEndPoint* mid = start + ((end - start) >> 1);
		if ( getEpValue(mid) < value )
		{
			start = mid;
		}
		else
		{
			end = mid;
		}
	}

	while( getEpValue(start) < value)
	{
		start++;
	}
	return start;
}


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::addNodePair( const hkpBpNode* n0, const hkpBpNode* n1, hkArray<hkpBroadPhaseHandlePair>& pairsOut, hkBool addMode) const
{
	if ( !n1->isMarker() )
	{
		if ( !n0->isMarker() )
		{
			hkpBroadPhaseHandlePair& pair = pairsOut.expandOne();
			pair.m_a = n0->m_handle;
			pair.m_b = n1->m_handle;
			return;
		}
		hkAlgorithm::swap(n0,n1);
	}
	HK_ASSERT( 0xf04edcf1, !n0->isMarker() );
	{
		int i0 = n0 - m_nodes.begin();
		hkpBpMarker& m = n1->getMarker( m_markers );
		if ( addMode )
		{
			m.m_overlappingObjects.pushBack(i0);
		}
		else
		{
			int i = m.m_overlappingObjects.indexOf( i0 );
			m.m_overlappingObjects.removeAt(i);
		}
	}
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::queryBatchAabbSub( const hkUint32* bitFieldOfQueryNodes, hkArray<hkpBroadPhaseHandlePair>& pairsOut, hkBool addMode ) const
{
	hkInplaceArray< BpInt,256 > openQueryNodes;
	hkInplaceArray< BpInt,256 > openOtherNodes;

	const hkpBpAxis& axis = m_axis[0];

	const hkpBpEndPoint* epEnd = &axis.m_endPoints.back();
    for ( const hkpBpEndPoint* ep = &axis.m_endPoints[1]; ep < epEnd; ep++)
	{
		int nodeIndex = ep->m_nodeIndex;
		int isQueryNode = staticIsBitSet( bitFieldOfQueryNodes, nodeIndex );

			//
			// if endpoint, remove from open interval lists
			//
		if ( ep->isMaxPoint())
		{
			hkInplaceArray< BpInt,256 >& array = (isQueryNode)? openQueryNodes: openOtherNodes;
			int index = array.indexOf( nodeIndex );
			array.removeAt( index );
			continue;
		}

			//
			//	Else add overlaps:
			//		openQueryNodes -- allNodes and
			//		openQueryNodes+openOtherNodes -- queryNodes
		const hkpBpNode* node = &m_nodes[nodeIndex];
		{
			for (int oq = 0; oq < openQueryNodes.getSize(); oq++)
			{
				BpInt oqi = openQueryNodes[oq];
				const hkpBpNode* oqn = &m_nodes[oqi];
				if ( !oqn->yzDisjoint( *node ) )
				{
					addNodePair( oqn, node, pairsOut, addMode );
				}
			}
		}

		if ( isQueryNode )
		{
			for (int oo = 0; oo < openOtherNodes.getSize(); oo++)
			{
				BpInt ooi = openOtherNodes[oo];
				const hkpBpNode* oon = &m_nodes[ooi];
				if ( !oon->yzDisjoint( *node ) )
				{
					addNodePair( oon, node, pairsOut, addMode );
				}
			}
			openQueryNodes.pushBack( nodeIndex );
		}
		else
		{
			openOtherNodes.pushBack( nodeIndex );
		}
	}
}
#endif

 const hkp3AxisSweep::hkpBpNode getNode( const hkp3AxisSweep::hkpBpNode* node, int i );

void hkp3AxisSweep::querySingleAabbSub( hkpBroadPhaseHandle* object, const hkUint32 *bitField, hkpBpNode& refNode, hkpBpQuerySingleType type,
									   hkArray<hkpBroadPhaseHandlePair>* pairs_out, hkArrayBase<const hkpBpNode*>* nodesOut, hkpBroadPhaseCastCollector* collector) const
{
	hkpBroadPhaseHandlePair pair;
	pair.m_a = object;

	const hkpBpNode *node = &m_nodes[0];
	const hkUint32* end = bitField + (m_nodes.getSize() >> 5) + 1;

	for( ; bitField < end; node += 32, bitField++ ) //check 32 at a time
	{
		hkUint32 mask = bitField[0];
		if ( mask==0 )
		{
			continue;
		}
		const hkpBpNode *n = node;
		do
		{
			if ( mask & 0xf)
			{
#ifndef HK_PLATFORM_SPU
				if ( (mask & 1) && !n[0].yzDisjoint( refNode ) && !n[0].isMarker() )
				{
					if ( type == HK_BP_REPORT_HANDLES)
					{
						pair.m_b = n[0].m_handle;
						pairs_out->pushBack( pair );
					}
					else if (type == HK_BP_REPORT_NODES)
					{
						nodesOut->pushBackUnchecked(n+0);
					}
					else
					{
						collector->addBroadPhaseHandle(n[0].m_handle, 0);
					}
				}
				if ( (mask & 2) && !n[1].yzDisjoint( refNode ) && !n[1].isMarker() )
				{
					if ( type == HK_BP_REPORT_HANDLES)
					{
						pair.m_b = n[1].m_handle;
						pairs_out->pushBack( pair );
					}
					else if (type == HK_BP_REPORT_NODES)
					{
						nodesOut->pushBackUnchecked(n+1);
					}
					else
					{
						collector->addBroadPhaseHandle(n[1].m_handle, 0);
					}
				}
				if ( (mask & 4) && !n[2].yzDisjoint( refNode ) && !n[2].isMarker() )
				{
					if ( type == HK_BP_REPORT_HANDLES)
					{
						pair.m_b = n[2].m_handle;
						pairs_out->pushBack( pair );
					}
					else if (type == HK_BP_REPORT_NODES)
					{
						nodesOut->pushBackUnchecked(n+2);
					}
					else
					{
						collector->addBroadPhaseHandle(n[2].m_handle, 0);
					}
				}
				if ( (mask & 8) && !n[3].yzDisjoint( refNode ) && !n[3].isMarker() )
				{
					if ( type == HK_BP_REPORT_HANDLES)
					{
						pair.m_b = n[3].m_handle;
						pairs_out->pushBack( pair );
					}
					else if (type == HK_BP_REPORT_NODES)
					{
						nodesOut->pushBackUnchecked(n+3);
					}
					else
					{
						collector->addBroadPhaseHandle(n[3].m_handle, 0);
					}
				}
#else
				for (int i=0; i<4; i++)
				{
					hkp3AxisSweep::hkpBpNode n_i = getNode(n, i);
					if ( (mask & (1<<i)) && !n_i.yzDisjoint( refNode ) && !n_i.isMarker() )
					{
						if ( type == HK_BP_REPORT_HANDLES)
						{
							pair.m_b = n_i.m_handle;
						pairs_out->pushBackUnchecked( pair );
						}
						else if (type == HK_BP_REPORT_NODES)
						{
							nodesOut->pushBackUnchecked(n+i);
					}
					else
					{
							collector->addBroadPhaseHandle(n_i.m_handle, 0);
						}
					}
				}
#endif
			}
			n += 4;
			mask = mask>>4;
		}
		while (mask);
	}
}


void hkp3AxisSweep::_querySingleAabb( const hkAabb& aabbIn, hkpBpQuerySingleType type, hkArray<hkpBroadPhaseHandlePair>* pairs_out, hkArrayBase<const hkpBpNode*>* nodesOut, hkpBroadPhaseCastCollector* collector) const
{
	HK_INTERNAL_TIMER_BEGIN_LIST("querySingleAabb", "marker" );

	hkpBroadPhaseHandle* object = HK_NULL;

	// clear the scratch area
	int numNodes = m_nodes.getSize();
	int bitFieldSize = (numNodes>>5)+8;
#if !defined(HK_PLATFORM_SPU)
	hkArray<hkUint32>::Temp bitfieldArea( bitFieldSize );
	hkUint32* bitField = bitfieldArea.begin();
#else
	hkUint32* bitField = hkAllocateStack<hkUint32>(bitFieldSize, "_querySingleAabbBitField");
#endif
	memclear16( bitField, numNodes>>3 );

	//AABB to integer
	hkAabbUint32 aabb;
	_convertAabbToInt( aabbIn, aabb );

	// run through the x axis and set the bits
	{
#ifndef HK_PLATFORM_SPU
		const hkpBpEndPoint* ep = &m_axis[0].m_endPoints[1];	// skip the first point
#else
		 hkSpuReadOnlyIterator<hkp3AxisSweep::hkpBpEndPoint, 256, hkSpuWorldGetClosestPointsDmaGroups::GET_BROADPHASE_AND_COMMANDS_AND_FILTER> ep;
		 ep.init( &m_axis[0].m_endPoints[1] );
#endif


#ifndef HK_PLATFORM_SPU
		// <ce.todo> Get this working after we fix markers :)
		if (getNumMarkers())
		{
			int x_pos = aabb.m_min[0];
			int pos = x_pos >> (HK_BP_NUM_VALUE_BITS-m_ld2NumMarkers);
			if ( pos > 0)
			{
				HK_ASSERT(0x55d2be4b,  pos <= m_numMarkers);
				hkpBpMarker* m = &m_markers[pos-1];
				{
					staticFlipBit( bitField, m->m_nodeIndex );
					BpInt* o = m->m_overlappingObjects.begin();
					for (int k = m->m_overlappingObjects.getSize()-1; k>=0; k--)
					{
						staticFlipBit( bitField, *o);
						o++;
					}
					const hkpBpNode& markerNode = m_nodes[ m->m_nodeIndex ];
					ep = &m_axis[0].m_endPoints[ markerNode.min_x + 1];

					// see setBitsBasedOnXInterval for next lines
					{
						const hkpBpEndPoint *end = &m_axis[0].m_endPoints[ markerNode.max_x ];
						while ( ep < end )
						{
							if ( !ep->isMaxPoint() )
							{
								staticClearBit( bitField, ep->m_nodeIndex);
							}
							ep++;
						}
					}
					ep = &m_axis[0].m_endPoints[ markerNode.min_x + 1];
				}
			}
		}
#endif // HK_PLATFORM_SPU
		while ( ep->m_value < aabb.m_min[0] )
		{
			staticFlipBit( bitField, ep->m_nodeIndex);
			ep++;
		}

		while ( ep->m_value < aabb.m_max[0] )
		{
			if ( !ep->isMaxPoint() )
			{
				staticFlipBit( bitField, ep->m_nodeIndex);
			}
			ep++;
		}
	}

	HK_INTERNAL_TIMER_SPLIT_LIST("yz-Axis");

	// set up the initial node min_x/y max_x/y
	hkpBpNode refNode;

	{
		const hkpBpAxis& axis = m_axis[1];
		const hkpBpEndPoint *sp = &axis.m_endPoints[0];
		refNode.min_y = axis.find( sp+1, axis.m_endPoints.end()-2, aabb.m_min[1]) - sp;
		refNode.max_y = axis.find( sp+1, axis.m_endPoints.end()-2, aabb.m_max[1]) - (sp + 1);
	}
	{
		const hkpBpAxis& axis = m_axis[2];
		const hkpBpEndPoint *sp = &axis.m_endPoints[0];
		refNode.min_z = axis.find( sp+1, axis.m_endPoints.end()-2, aabb.m_min[2]) - sp;
		refNode.max_z = axis.find( sp+1, axis.m_endPoints.end()-2, aabb.m_max[2]) - (sp + 1);
	}

	HK_INTERNAL_TIMER_SPLIT_LIST("ScanBitfield");

	// run through the bits and search the nodes
	querySingleAabbSub( object, bitField, refNode, type, pairs_out, nodesOut, collector);

#if defined(HK_PLATFORM_SPU)
	hkDeallocateStack(bitField);
#endif

	HK_INTERNAL_TIMER_END_LIST();
}


void hkp3AxisSweep::querySingleAabb( const hkAabb& aabbIn, hkArray<hkpBroadPhaseHandlePair>& pairs_out) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	_querySingleAabb( aabbIn, HK_BP_REPORT_HANDLES, &pairs_out, HK_NULL, HK_NULL);
}

void hkp3AxisSweep::querySingleAabbWithCollector( const hkAabb& aabbIn, hkpBroadPhaseCastCollector* collector) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	_querySingleAabb( aabbIn, HK_BP_COLLECTOR, HK_NULL, HK_NULL, collector);
}


#if !defined(HK_PLATFORM_SPU)
int hkp3AxisSweep::getAabbCacheSize() const
{
	int maxNumNodes = m_nodes.getSize() - getNumMarkers();
	int maxNumEndPoints = maxNumNodes * 2;

		 // header info
	int size = sizeof(hkpBpAxis) * 3;

		// the array data
	size += hkSizeOf(hkpBpEndPoint)  * 3 * maxNumEndPoints;
	return size;
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::calcAabbCache( const hkAabb& aabb, hkpBroadPhaseAabbCache* AabbCacheOut) const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );

	hkArray<const hkpBpNode*>::Temp overlaps;
	overlaps.reserveExactly( m_nodes.getSize() );
	_querySingleAabb( aabb, HK_BP_REPORT_NODES, HK_NULL, &overlaps, HK_NULL );

	HK_ASSERT(0x72dbfa0e,  overlaps.getSize() < m_nodes.getSize() - getNumMarkers());

	calcAabbCacheInternal(overlaps, AabbCacheOut);
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::calcAabbCache( const hkArrayBase<hkpCollidable*>& overlappingCollidables, hkpBroadPhaseAabbCache* AabbCacheOut) const
{
	hkLocalArray<const hkpBpNode*> overlaps( overlappingCollidables.getSize() );

	for (int i=0; i<overlappingCollidables.getSize(); i++)
	{
		hkUint32 nodeIndex = overlappingCollidables[i]->getBroadPhaseHandle()->m_id;
		overlaps.pushBackUnchecked( &m_nodes[ nodeIndex ] );
	}

	HK_ASSERT(0x72dbfa0e,  overlaps.getSize() < m_nodes.getSize() - getNumMarkers());

	calcAabbCacheInternal(overlaps, AabbCacheOut);
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::calcAabbCacheInternal( const hkArrayBase<const hkpBpNode*>& overlaps, hkpBroadPhaseAabbCache* AabbCacheOut) const
{
	hkpBpAxis* axis = reinterpret_cast<hkpBpAxis*>(AabbCacheOut);
	{
		hkpBpEndPoint *ep = reinterpret_cast<hkpBpEndPoint*>(axis+3);

		int size = overlaps.getSize()*2+2;
		new( axis+0 ) hkpBpAxis(ep + 0 * size,size);
		new( axis+1 ) hkpBpAxis(ep + 1 * size,size);
		new( axis+2 ) hkpBpAxis(ep + 2 * size,size);
	}

	hkpBpEndPoint ep;
	ep.m_value = 0;
	ep.m_nodeIndex = 0;

	axis[0].m_endPoints.pushBackUnchecked( ep );
	axis[1].m_endPoints.pushBackUnchecked( ep );
	axis[2].m_endPoints.pushBackUnchecked( ep );

	hkpBpNode** p = const_cast<hkpBpNode**>(overlaps.begin());

	{
		for (int i = overlaps.getSize()-1; i>=0; i--)
		{
			axis[0].m_endPoints.pushBackUnchecked( m_axis[0].m_endPoints[ (*p)->_getMin<0>() ] );
			axis[0].m_endPoints.pushBackUnchecked( m_axis[0].m_endPoints[ (*p)->_getMax<0>() ] );
			axis[1].m_endPoints.pushBackUnchecked( m_axis[1].m_endPoints[ (*p)->_getMin<1>() ] );
			axis[1].m_endPoints.pushBackUnchecked( m_axis[1].m_endPoints[ (*p)->_getMax<1>() ] );
			axis[2].m_endPoints.pushBackUnchecked( m_axis[2].m_endPoints[ (*p)->_getMin<2>() ] );
			axis[2].m_endPoints.pushBackUnchecked( m_axis[2].m_endPoints[ (*p)->_getMax<2>() ] );
			p++;
		}
	}
	{
		for (int i = 0; i < 3; i++)
		{
			hkSort( axis[i].m_endPoints.begin()+1, axis[i].m_endPoints.getSize()-1);
		}
	}

	ep.m_value = BpInt(AABB_MAX_VALUE);
	ep.m_nodeIndex = 0;
	axis[0].m_endPoints.pushBackUnchecked( ep );
	axis[1].m_endPoints.pushBackUnchecked( ep );
	axis[2].m_endPoints.pushBackUnchecked( ep );
}
#endif

template <typename T>
struct ValueIntPair
{
	T m_value;
	T m_oldIndex;

	hkBool operator <(const ValueIntPair &b) const
	{
		return m_value < b.m_value;
	}
};


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::defragment()
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	CHECK_CONSISTENCY();
	int size = m_nodes.getSize();
	hkArray<hkpBpNode>::Temp oldNodes(size);
	hkArray< ValueIntPair<hkp3AxisSweep::BpInt> >::Temp newToOld(size);
	hkArray<int>::Temp oldToNew(size);

	//
	//	copy the nodes
	//
	{
		for (int i=0; i < m_nodes.getSize(); i++)
		{
			oldNodes[i] = m_nodes[i];
			newToOld[i].m_value = m_nodes[i].min_x;
			newToOld[i].m_oldIndex = i;
		}
	}

	hkSort( &newToOld[1], size-1 ); // do not sort the first node

	//
	// build the reverse mapping and resort all the nodes
	//
	{
		for (int i=0; i < m_nodes.getSize(); i++)
		{
			int newIndex = newToOld[i].m_oldIndex;
			oldToNew[ newIndex ] = i;
			m_nodes[ i ] = oldNodes[ newIndex ];
		}
	}

	// update reverse pointers
	{
		for (int i=1; i < m_nodes.getSize(); i++)
		{
			hkpBpNode& node = m_nodes[ i ];
			if ( !node.isMarker() )
			{
				m_nodes[ i ].m_handle->m_id = i;	// fix the reverse mapping
			}
			else
			{
				node.getMarker(m_markers).m_nodeIndex = i;
			}
		}
	}

	//
	//	Update all markers
	//
	{
		for (int i = 0; i < getNumMarkers(); i++)
		{
			hkpBpMarker& m = m_markers[i];


			for (int e = m.m_overlappingObjects.getSize()-1; e>=0; e--)
			{
				m.m_overlappingObjects[e] = oldToNew[ m.m_overlappingObjects[e]];
			}
		}
	}

	//
	// update all the ep
	//
	{
		for (int a = 0; a < 3; a++)
		{
			hkpBpAxis &axis = m_axis[a];
			hkpBpEndPoint *ep = axis.m_endPoints.begin();
			for (int i=0; i < axis.m_endPoints.getSize(); i++)
			{
				ep[i].m_nodeIndex = oldToNew[ ep[i].m_nodeIndex ];
			}
		}
	}


	{
		for (int i = 1; i < m_nodes.getSize(); i++)
		{
			_fixDeterministicOrderAfterNodeIdWasDecreased( i );
		}
	}


	CHECK_CONSISTENCY();
}
#endif

// Optimized version to remove LHS when converting int in ep->m_value to float - performance increase of ~10-15%
inline void calcCurDist( int component, hkVector4& curDistance, const hkp3AxisSweep::hkpBpEndPoint* ep, hkVector4& invScale, hkVector4& invOffset)
{
#ifdef HK_PLATFORM_XBOX360
	static HK_ALIGN16( const hkUint32 compmask[3][4] ) = {
		{ 0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000 },
		{ 0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000 },
		{ 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000 }
	};

#  ifdef HK_BROADPHASE_32BIT
	// Load 32 bits
	__vector4 val =  __vor( __lvlx( ep, 0 ), __lvrx( ep, 16 ) ); // val(0) = ep->m_value
#  else
	// Load 16 bits
	static HK_ALIGN16( const hkUint32 mask[4] ) = { 0x0000FFFF, 0x00000000, 0x00000000, 0x00000000 };
	__vector4 val =  __vor( __lvlx( ep, -2 ), __lvrx( ep, 16-2 ) );
	val = __vand( val, *(__vector4*)mask ); // val(0) = ep->m_value
#  endif
	val = __vcfsx(val, 0);	// val = float(ep->value)
	val = __vspltw( val, 0);
	val = __vsubfp( __vmulfp(val, invScale.m_quad), invOffset.m_quad);
	curDistance.m_quad = __vsel( curDistance.m_quad, val, *(__vector4*)compmask[component] );

#elif defined(HK_PLATFORM_PS3_PPU)

	static HK_ALIGN16( const hkUint32 compmask[3][4] ) = {
		{ 0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000 },
		{ 0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000 },
		{ 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000 }
	};
#  ifdef HK_BROADPHASE_32BIT
	// Load 32 bits
	vector unsigned int val =  (vector unsigned int) vec_or( vec_lvlx( 0, &ep->m_value ), vec_lvrx( 16, &ep->m_value ) ); // val(0) = ep->m_value
#  else
	// Load 16 bits
	static HK_ALIGN16( const hkUint32 shifts[4] ) = { 0x0000010, 0x00000000, 0x00000000, 0x00000000 };
	vector unsigned int val = (vector unsigned int) vec_or( vec_lvlx( 0, &ep->m_value ), vec_lvrx( 16, &ep->m_value ) );
	val = vec_sr( val, *(vector unsigned int*)shifts ); // val(0) = ep->m_value
#  endif
	vector float fval;
	fval = vec_ctf(val, 0);  // val = float(ep->value)
	fval = vec_splat( fval, 0);
	fval = vec_sub( vec_mul(fval, invScale.m_quad), invOffset.m_quad);
	curDistance.m_quad = vec_sel( curDistance.m_quad, fval, *(vector unsigned int*)compmask[component] );

#else

	curDistance(component) = ep->m_value * invScale(component) - invOffset(component);

#endif

}

// Optimized version to remove LHS  - performance increase of ~10% 
inline int getMinComponent( const hkVector4& curDistance, hkReal& min )
{
#ifdef HK_PLATFORM_XBOX360

	__vector4 v = curDistance.m_quad;

	static HK_ALIGN16( const hkUint32 index[4] ) = { 0x0, 0x1, 0x2, 0x0 };
	__vector4 i = *(__vector4*)index;

	__vector4 vr1 = __vpermwi( v, VPERMWI_CONST( 1, 2, 0, 0 ) );
	__vector4 ir1 = __vpermwi( i, VPERMWI_CONST( 1, 2, 0, 0 ) );
	__vector4 vr2 = __vpermwi( v, VPERMWI_CONST( 2, 0, 1, 0 ) );
	__vector4 ir2 = __vpermwi( i, VPERMWI_CONST( 2, 0, 1, 0 ) );

	__vector4 v_lt_vr1_m = __vcmpgefp( v, vr1 );
	__vector4 v_lt_vr1   = __vsel( v, vr1, v_lt_vr1_m );
	__vector4 i_lt_ir1   = __vsel( i, ir1, v_lt_vr1_m );

	__vector4 v_lt_vr1_lt_vr2_m = __vcmpgefp( v_lt_vr1, vr2 );
	__vector4 v_lt_vr1_lt_vr2   = __vsel( v_lt_vr1, vr2, v_lt_vr1_lt_vr2_m );
	__vector4 i_lt_ir1_lt_ir2   = __vsel( i_lt_ir1, ir2, v_lt_vr1_lt_vr2_m );

	v_lt_vr1_lt_vr2 = __vspltw( v_lt_vr1_lt_vr2, 0);
	i_lt_ir1_lt_ir2 = __vspltw( i_lt_ir1_lt_ir2, 0);

	__stvewx( v_lt_vr1_lt_vr2, &min, 0);

	int minIndex;
	__stvewx( i_lt_ir1_lt_ir2, &minIndex, 0);
	return minIndex;

#elif defined(HK_PLATFORM_PS3_PPU)

	hkQuadReal v = curDistance.m_quad;

	const vector unsigned int i = (vector unsigned int){ 0x0, 0x1, 0x2, 0x0 };

	vector float vr1 = vec_perm( v, v, VPERMWI_CONST( Y, Z, X, X ) );
	vector unsigned int ir1 = vec_perm( i, i, VPERMWI_CONST( Y, Z, X, X ) );
	vector float vr2 = vec_perm( v, v, VPERMWI_CONST( Z, X, Y, X ) );
	vector unsigned int ir2 = vec_perm( i, i, VPERMWI_CONST( Z, X, Y, X ) );

	vector unsigned int v_lt_vr1_m = (vector unsigned int) vec_cmpge( v, vr1 );
	vector float v_lt_vr1   = vec_sel( v, vr1, v_lt_vr1_m );
	vector unsigned int i_lt_ir1   = vec_sel( i, ir1, v_lt_vr1_m );

	vector unsigned int v_lt_vr1_lt_vr2_m = (vector unsigned int) vec_cmpge( v_lt_vr1, vr2 );
	vector float v_lt_vr1_lt_vr2   = vec_sel( v_lt_vr1, vr2, v_lt_vr1_lt_vr2_m );
	vector unsigned int i_lt_ir1_lt_ir2   = vec_sel( i_lt_ir1, ir2, v_lt_vr1_lt_vr2_m );

	v_lt_vr1_lt_vr2 = vec_splat( v_lt_vr1_lt_vr2, 0);
	i_lt_ir1_lt_ir2 = vec_splat( i_lt_ir1_lt_ir2, 0);

	vec_ste( v_lt_vr1_lt_vr2, 0, &min);

	int minIndex;
	vec_ste( (vector signed int) i_lt_ir1_lt_ir2, 0, &minIndex);
	return minIndex;

#else
	int minDist = curDistance.getIndexOfMinComponent<3>();
	min = curDistance(minDist);
	return minDist;

		// those next lines give horrible code on all Platforms which have no native min2 and fselectGreaterEqualZero implementation
//	min = hkMath::min2( hkMath::min2( curDistance(0), curDistance(1) ) , hkMath::min2( curDistance(1) , curDistance(2)) );
//	return  hkMath::hkToIntFast( hkMath::fselectGreaterEqualZero( curDistance(0) - curDistance(1), hkMath::fselectGreaterEqualZero( curDistance(1) - curDistance(2), 2.0f, 1.0f ) , hkMath::fselectGreaterEqualZero( curDistance(0) - curDistance(2), 2.0f, 0.0f ) ) );

#endif
}

#ifdef HK_PLATFORM_SPU

extern hkSpu4WayCache* g_SpuCollideUntypedCache;

const hkp3AxisSweep::hkpBpNode getNode( const hkp3AxisSweep::hkpBpNode* node, int i )
{
	const hkp3AxisSweep::hkpBpNode* n = hkGetArrayElemUsingCache( node, i, g_SpuCollideUntypedCache, HK_SPU_UNTYPED_CACHE_LINE_SIZE);
	return *n;
}

#endif

//
// Fast Raycast through the broadphase using 3dda.
//
// The basic idea is that we use a walking algorithm:
//
//		From from to to. So we need to implement 2 steps:
//			1. Calculate the starting point
//			2. Traverse
//
// Concerning StartPoint:
//		The idea for the starting point is that we need to calculate all existing
//		AABBs, which do overlap our starting position. We use the same algorithm
//		as querySingleAabb for this task: Traverse one axis and set a 1 for every node
//		which overlaps on this axis. Then we look for 1s in this array and check
//		for overlaps on the other two axis
//
// Bits Used:
//     We use the lower 4 bits for the start point data and the upper 4 bits for temporary traversal
//
//	Concerning Traverse:
//		First we calculate a projection from broadphase integer space into raycast space:
//		In raycast space the ray goes from (0,0,0) to (1,1,1). If the ray is flat in one direction
//	    than the start and end value is set to 2 (effectively disabling further checks)
//		We keep three pointers (one for each axis) pointing to the next cell boundary to cross.
//		We find the closest border and cross it.
//		Whenever we cross a border, we also enter a new cell. So we simply update our original bitfield
//		and check whether we are fully inside an object.
//
//	Numerical Issues
//			Rules:
//				A) For every AABB in the broadphase, each of its original floating point min or max
//				   converted to int space should overlap with its int AABB.
//				B) For every AABB in the broadphase, each of its integer min or max taken and
//				   converted to original float space should overlap with its original float space AABB.
//
//			In other words, converting back and from int space should not change the result of a '<=' operation
//			So we need to make sure that the following equations are correct:
//
//			Given:  min <= x <= max
//					function fint(x) = int( floor( x ))
//
//			Than:   fint(min) <= fint(x) <= fint(max)		-> That means converting to int space does not change order of things
//			And:    fint(min) <= x <= fint(max) + 1			-> Thats the reason all AABBs max value use the
//															    m_offsetHigh == m_offsetLow+1
//			Unfortunately we are using int() instead of fint().
//			There is a number c (intCorrection), for which:
//					fint(y) = int(y+c);
//				->	fint(y-c) = int(y)
//
//				-> fint(min-c) <= x-c <= fint(max-c) + 1
//				->  int(min)   <= x-c <=  int(max) + 1			-> bingo: If we use c and add 1 to the max than our order should be ok
//																( +1 is already added, so we just have to worry about c)
void hkp3AxisSweep::castRay( const hkpCastRayInput& rayInput, hkpBroadPhaseCastCollector* collectorBase, int collectorStriding )    const
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	HK_ASSERT2(0x46e24201, rayInput.m_from.isOk<3>(), "hkp3AxisSweep::castRay has invalid 'from' vector." );
	HK_ASSERT2(0xaf542fe8, m_scale.length<4>().getReal() != 0, "Make sure to call set32BitOffsetAndScale() after creating the broadphase.");

	//
	// convert into integer space
	//
	HK_ALIGN16( hkUint32 aabb[4] );
	{
		convertVectorToInt( rayInput.m_from, aabb );
	}


	//
	// clear the scratch area, the scratch area is used to hold bits z = 4 y =2 x = 1
	//
	HK_INTERNAL_TIMER_BEGIN_LIST( "hkp3AxisSweep", "memory" );

#ifndef HK_PLATFORM_SPU
	hkArray<unsigned char>::Temp scratchArea(m_nodes.getSize()+16);
#else
	hkLocalBuffer<unsigned char> scratchArea(m_nodes.getSize()+16);
#endif
	{
		memclear16( scratchArea.begin(), m_nodes.getSize());		// todo optimize
	}
	HK_INTERNAL_TIMER_SPLIT_LIST( "bitfield" );

	//
	// run through the 3 axis, find the start cell (ep[3]) and set the bits (scratchArea)
	//
	const hkpBpEndPoint* startEp[3];

	//
	//	Get the axis info
	//
	const hkpBpAxis* axisArray = ( rayInput.m_aabbCacheInfo )? reinterpret_cast<const hkpBpAxis*>(rayInput.m_aabbCacheInfo) : &m_axis[0];
	{
		{
			unsigned char* bitField = scratchArea.begin();
			int mask = 0x11;
			for (int i=0;i<3;i++)
			{
				const hkpBpAxis& axis = axisArray[i];

#ifndef HK_PLATFORM_SPU

				// check whether we should walk from left or right
				const hkpBpEndPoint* p;
				const hkpBpEndPoint* midP = &axis.m_endPoints[axis.m_endPoints.getSize()>>1];
				if ( aabb[i] < midP->m_value )
				{
					//
					//	walk from left
					//
					const hkpBpEndPoint* endP = axis.m_endPoints.begin() + (axis.m_endPoints.getSize()-4);

					for( p = &axis.m_endPoints[1];  p < endP && p[3].m_value <= aabb[i]; p += 4 )
					{
						bitField[ p[0].m_nodeIndex ] ^= mask;
						bitField[ p[1].m_nodeIndex ] ^= mask;
						bitField[ p[2].m_nodeIndex ] ^= mask;
						bitField[ p[3].m_nodeIndex ] ^= mask;
					}

					for ( ;p->m_value <= aabb[i]; p++ )
					{
						bitField[ p->m_nodeIndex ] ^= mask;
					}
				}
				else
				{
					//
					//	walk from right
					//

					const hkpBpEndPoint* endP = axis.m_endPoints.begin()+4;

					for( p = &axis.m_endPoints[axis.m_endPoints.getSize()-2]; p >= endP && p[-3].m_value > aabb[i]; p -= 4 )
					{
						bitField[ p[ 0].m_nodeIndex ] ^= mask;
						bitField[ p[-1].m_nodeIndex ] ^= mask;
						bitField[ p[-2].m_nodeIndex ] ^= mask;
						bitField[ p[-3].m_nodeIndex ] ^= mask;
					}

					for ( ;p->m_value > aabb[i]; p--)
					{
						bitField[ p->m_nodeIndex ] ^= mask;
					}
					p++;	// we moved one to far
				}
#else
				// check whether we should walk from left or right
				const hkpBpEndPoint* midP = hkGetArrayElemUsingCache( axis.m_endPoints.begin(), axis.m_endPoints.getSize()>>1, g_SpuCollideUntypedCache, HK_SPU_UNTYPED_CACHE_LINE_SIZE );

				hkSpuReadOnlyIterator<hkp3AxisSweep::hkpBpEndPoint, 256, hkSpuWorldRayCastDmaGroups::GET_BROADPHASE> p;

				if ( aabb[i] < midP->m_value )
				{
					const hkp3AxisSweep::hkpBpEndPoint* endP = axis.m_endPoints.begin() + axis.m_endPoints.getSize();
					p.init( &axis.m_endPoints[1] );
					for ( ; p < endP &&  p->m_value <= aabb[ HK_HINT_SIZE16(i) ]; p++ )
					{
						bitField[ HK_HINT_SIZE16(p->m_nodeIndex) ] ^= mask;
					}
				}
				else
				{
					const hkp3AxisSweep::hkpBpEndPoint* endP = axis.m_endPoints.begin();
					p.init ( &axis.m_endPoints[axis.m_endPoints.getSize()-2] );
					for ( ; p >= endP &&  p->m_value > aabb[ HK_HINT_SIZE16(i) ]; p-- )
					{
						bitField[ HK_HINT_SIZE16(p->m_nodeIndex) ] ^= mask;
					}
					p++;
				}
#endif

				startEp[i] = p;
				mask = mask << 1;
			}
		}
	}

	//
	//	Search for start overlaps
	//
	HK_INTERNAL_TIMER_SPLIT_LIST( "StartOverlaps");

	hkLocalBuffer <hkReal> minHitFractions( rayInput.m_numCasts );
	{
		for (int x = 0; x < rayInput.m_numCasts ; x++){ minHitFractions[x] = 1.0f;}
	}

	{
		const hkpBpNode *node = &m_nodes[0];
		int numNodes = m_nodes.getSize();

		const int* bitsInt = reinterpret_cast<const int *>( scratchArea.begin() );
		const int* end = bitsInt + (numNodes>>2) + 1;
		for( ; bitsInt < end;  ) //check four at a time
		{
			if ( ((bitsInt[0] + 0x01010101 ) & 0x08080808) ==0 )
			{
				if ( ((bitsInt[1] + 0x01010101 ) & 0x08080808) ==0 )
				{
					if ( ((bitsInt[2] + 0x01010101 ) & 0x08080808) ==0 )
					{
						node += 12;
						bitsInt+=3;
						continue;
					}
					node += 8;
					bitsInt+=2;
					continue;
				}
				node += 4;
				bitsInt++;
				continue;
			}

			const unsigned char* bitChar = reinterpret_cast<const unsigned char*>(bitsInt);
#define DO_HIT(NODE)		{															\
						    hkpBroadPhaseCastCollector* collector = collectorBase;	\
							for (int x = 0; x < rayInput.m_numCasts ; x++)			\
							{														\
								minHitFractions[x] = hkMath::min2(minHitFractions[x] , collector->addBroadPhaseHandle(NODE.m_handle,x));	\
							    collector = hkAddByteOffset( collector, collectorStriding );	\
							}														\
						};
#ifndef HK_PLATFORM_SPU
			if ( bitChar[0]==0x77 && !node[0].isMarker())		DO_HIT(node[0]);
			if ( bitChar[1]==0x77 && !node[1].isMarker())		DO_HIT(node[1]);
			if ( bitChar[2]==0x77 && !node[2].isMarker())		DO_HIT(node[2]);
			if ( bitChar[3]==0x77 && !node[3].isMarker())		DO_HIT(node[3]);
#else
			hkp3AxisSweep::hkpBpNode bpNode;
			bpNode = getNode( node, 0 ); if ( bitChar[0]==0x77 && !bpNode.isMarker())		DO_HIT(bpNode);
			bpNode = getNode( node, 1 ); if ( bitChar[1]==0x77 && !bpNode.isMarker())		DO_HIT(bpNode);
			bpNode = getNode( node, 2 ); if ( bitChar[2]==0x77 && !bpNode.isMarker())		DO_HIT(bpNode);
			bpNode = getNode( node, 3 ); if ( bitChar[3]==0x77 && !bpNode.isMarker())		DO_HIT(bpNode);
#endif
			node += 4;
			bitsInt++;
#undef DO_HIT
		}
		scratchArea.begin()[0] = 0x88;
	}

	HK_INTERNAL_TIMER_SPLIT_LIST( "Walk");

	hkpBroadPhaseCastCollector* collector = collectorBase;
	int numNodes = m_nodes.getSize();
	for (int castIndex = 0; castIndex < rayInput.m_numCasts; collector = hkAddByteOffset( collector, collectorStriding ), castIndex++)
	{
		hkReal minHitFraction = minHitFractions[castIndex];

#ifndef HK_PLATFORM_SPU
		const hkpBpEndPoint* ep[3];
		ep[0] = startEp[0];
		ep[1] = startEp[1];
		ep[2] = startEp[2];
#else
		hkSpuReadOnlyIterator<hkp3AxisSweep::hkpBpEndPoint, 256,  hkSpuWorldRayCastDmaGroups::GET_BROADPHASE> ep[3];
		ep[0].init( startEp[0] );
		ep[1].init( startEp[1] );
		ep[2].init( startEp[2] );
#endif


		//
		//	Initialize the invScale and offset in a way, so that the ray
		//  goes from 0,0,0 to 1,1,1. If the ray is flat in one direction
		//  than the start and end value is set to 2
		//
		hkVector4 invScale;
		hkVector4 invOffset;
		hkLong direction[3];
		{
			hkVector4 to = *hkAddByteOffsetConst( rayInput.m_toBase, rayInput.m_toStriding * castIndex);
			HK_ASSERT2(0x72865bd8, to.isOk<3>(), "hkp3AxisSweep::castRay has invalid 'to' vector.");
#if !defined(HK_PLATFORM_SPU)
#if defined HK_DEBUG
			hkVector4 tolerence; tolerence.setAll(0.00001f);
			hkAabb debugAabb;
			getExtents(debugAabb.m_min, debugAabb.m_max);
			debugAabb.m_min.sub(tolerence);
			debugAabb.m_max.add(tolerence);
			if (!debugAabb.containsPoint(to))
			{
				HK_WARN_ONCE(0x38f1276d, "Raycast target is outside the broadphase. False misses may be reported for objects that are outside the broadphase.");
			}
#endif //defined HK_DEBUG
#endif
			hkVector4 dir; dir.setSub( to, rayInput.m_from );
			hkVector4 delta; delta.setMul(m_scale, dir);
			hkVector4 fabsDelta; fabsDelta.setAbs( delta );
			hkVector4 invDelta; invDelta.setReciprocal(delta);
			hkVector4 rayStartInBroadPhaseSpace; rayStartInBroadPhaseSpace.setAdd(rayInput.m_from, m_offsetLow); rayStartInBroadPhaseSpace.mul(m_scale);
			hkVector4 rayEndInBroadPhaseSpace; rayEndInBroadPhaseSpace.setAdd(to, m_offsetLow); rayEndInBroadPhaseSpace.mul(m_scale);

			hkVector4Comparison comp;
			{
				const hkSimdReal rEps = hkSimdReal::getConstant<HK_QUADREAL_EPS>();
				hkVector4 scaledStart; 
					scaledStart.setMul(rEps, rayStartInBroadPhaseSpace);
					scaledStart.setAbs(scaledStart);
				hkVector4 scaledEnd; 
					scaledEnd.setMul(rEps, rayEndInBroadPhaseSpace);
					scaledEnd.setAbs(scaledEnd);
				const hkVector4Comparison absDeltaLessStart = fabsDelta.less(scaledStart);
				const hkVector4Comparison absDeltaLessEnd   = fabsDelta.less(scaledEnd);
				comp.setOr(absDeltaLessStart, absDeltaLessEnd);
			}

			invScale = invDelta;
			{
				hkVector4 i2fCorrection; i2fCorrection.setAll(m_intToFloatFloorCorrection);
				hkVector4 correctedPosition; correctedPosition.setSub(rayStartInBroadPhaseSpace, i2fCorrection);
				invOffset.setMul(correctedPosition, invDelta);
			}

			for (int i=0; i<3; i++)
			{
				//
				//	Check for allowed division
				//
				if (comp.anyIsSet(hkVector4Comparison::getMaskForComponent(i)))
				{
					invScale.zeroComponent(i);
					invOffset(i) = -2.0f;
					continue;
				}

				direction[i] = hkSizeOf( hkpBpEndPoint );
				if ( delta.getComponent(i).isLessZero() )
				{
					direction[i] = - hkSizeOf( hkpBpEndPoint );
					ep[i]--;	// if we walk left, the next boundary is left
				}
			}
		}

		//
		// test start point (see docu at start of this function
		//
	#if defined(HK_DEBUG) && !defined( HK_BROADPHASE_32BIT)
		if (1)
		{
			for (int i = 0; i <3; i++ )
			{
				if ( invScale(i) == 0.0f  ) { continue; }
				if ( aabb[i] == AABB_MIN_VALUE ) { continue; }
				if ( aabb[i] >= AABB_MAX_VALUE-1 ) { continue; }

				// test
				hkReal testMin =  aabb[i     ] * invScale(i) - invOffset(i);
				hkReal testMax = (aabb[i] + 1) * invScale(i) - invOffset(i);
				if ( direction[i] < 0 ) { testMin *= -1.0f;	testMax *= -1.0f; }
				const hkReal fltEpsilon = 1.192092896e-07f;
				hkReal numericalTolerance = hkMath::fabs( invOffset(i) ) * fltEpsilon * 16.0f;
				HK_ASSERT2(0x2e4b0cdb,  testMin <= numericalTolerance && testMax >= -numericalTolerance, "Numerical inconsistency in hkp3AxisSweep" );
			}
		}
	#endif

		hkVector4 curDistance;
		{
			hkVector4 epVals; epVals.set(ep[0]->m_value, ep[1]->m_value, ep[2]->m_value);
			curDistance.setMul(epVals, invScale);
			curDistance.sub(invOffset);
		}

		unsigned char* bitField = scratchArea.begin();

		while(1) {

			// find closest cell boundary
			hkReal minComponent;
			int component = getMinComponent( curDistance, minComponent );

			//	Early out if distance is too big
			if ( minComponent > minHitFraction )
			{
				break;
			}

			while(1)	// quick loop
			{
				//
				// Update bitfield and check if we entered an object
				//
				{
					const int nodeindex = ep[component]->m_nodeIndex;
					bitField[nodeindex] ^= BIT_SHIFT_LEFT_1( component + 4 ); // 0x10<<component

					if (bitField[nodeindex] >= 0x70)
					{
						if ( nodeindex == 0 )
						{
							// just encountered the boundary object, so do not walk into that direction any longer
							curDistance(component) = 2.0f;
							break; // quick loop
						}

						//entering an AABB
#ifndef HK_PLATFORM_SPU
						const hkpBpNode& node = m_nodes[nodeindex];
#else
						const hkpBpNode node = getNode( m_nodes.begin(), nodeindex);
#endif
						if ( !node.isMarker())
						{
							minHitFraction = hkMath::min2(minHitFraction , collector->addBroadPhaseHandle(node.m_handle, castIndex));
						}
					}
				}

				//
				//	walk: quick check, whether we really moved a distance forward, if no, quickloop
				//  else recalculate the distances and the min component
				//
				{
					BpInt lastValue = ep[component]->m_value;
#ifndef HK_PLATFORM_SPU
					ep[component] = hkAddByteOffsetConst<hkpBpEndPoint>(ep[component], direction[component] );
#else
					if (direction[component] > 0)
					{
						ep[component]++;
					}
					else
					{
						ep[component]--;
					}
#endif
					BpInt newValue = ep[component]->m_value;
					if ( lastValue == newValue )
					{
						continue; // quick loop
					}
					hkpBpEndPoint endPt = *ep[component];
					calcCurDist( component, curDistance, &endPt, invScale, invOffset );
					break;
				}
			}
		}
		if ( castIndex < rayInput.m_numCasts-1)
		{
			//
			//	Reuse bitfield by copying the startpoint data to the temporary data
			//
			int* bitsInt = reinterpret_cast<int *>( scratchArea.begin() );
			const int* end = bitsInt + (numNodes>>2) + 1;
			int mask = 0x0f0f0f0f;
			for( ; bitsInt < end;  )
			{
				int a = bitsInt[0];
				int b = bitsInt[1];
				a &= mask;
				b &= mask;
				a |= a << 4;
				b |= b << 4;
				bitsInt[0] = a;
				bitsInt[1] = b;
				bitsInt+=2;
			}
		}
	}
	HK_INTERNAL_TIMER_END_LIST();
}

//
// cast Aabb
//
void hkp3AxisSweep::castAabb( const hkpCastAabbInput& input, hkpBroadPhaseCastCollector& collector )    const
{
	HK_ASSERT2(0xaf542fe3, m_scale.length<4>().getReal() != 0, "Make sure to call set32BitOffsetAndScale() after creating the broadphase.");

	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );
	//
	// convert into integer space
	//
	HK_ALIGN16( hkUint32 aabbMin[4] );
	HK_ALIGN16( hkUint32 aabbMax[4] );


	hkVector4 fromMin; fromMin.setSub( input.m_from, input.m_halfExtents );
	hkVector4 fromMax; fromMax.setAdd( input.m_from, input.m_halfExtents );
	{
		convertVectorToInt( fromMin, aabbMin );
		convertVectorToInt( fromMax, aabbMax );
	}

#ifdef HK_DEBUG
	{
		hkVector4 toMin; toMin.setSub(input.m_to, input.m_halfExtents); 
		hkVector4 toMax; toMax.setAdd(input.m_to, input.m_halfExtents); 

		const bool outside =   !(m_aabb.m_min.allLess<3>(fromMin) && fromMax.allLess<3>(m_aabb.m_max)
							&&   m_aabb.m_min.allLess<3>(  toMin) &&   toMax.allLess<3>(m_aabb.m_max) );

		if ( outside )
		{
			HK_WARN_ONCE(0xad367291, "Linear cast is partially outside of broadphase. False misses may be reported for objects that are outside the broadphase." );
		}
	}
	
#endif

	//
	// clear the scratch area, the scratch area is used to hold bits z = 4 y =2 x = 1
	//
	HK_INTERNAL_TIMER_BEGIN_LIST( "hkp3AxisSweep", "bitfield" );


#ifndef HK_PLATFORM_SPU
	hkArray<unsigned char>::Temp scratchArea(m_nodes.getSize()+16);
#else
	hkLocalBuffer<unsigned char> scratchArea(m_nodes.getSize()+16);
#endif
	{
		memclear16( scratchArea.begin(), m_nodes.getSize());
	}
	
	//
	// run through the 3 axis, find the start cell (ep[3]) and set the bits (scratchArea)
	//
	const hkpBpEndPoint* startEpMin[3];
	const hkpBpEndPoint* startEpMax[3];
	const hkpBpAxis* axisArray = ( input.m_aabbCacheInfo )? reinterpret_cast<const hkpBpAxis*>(input.m_aabbCacheInfo) : &m_axis[0];
	{
		{
			unsigned char* bitField = scratchArea.begin();
			int mask = 1;
			for (int i=0;i<3;i++)
			{
				const hkpBpAxis& axis = axisArray[i];
#ifndef HK_PLATFORM_SPU
				// check whether we should walk from left or right
				const hkpBpEndPoint* p;
				const hkpBpEndPoint* midP = &axis.m_endPoints[axis.m_endPoints.getSize()>>1];
				if ( aabbMin[i] < midP->m_value )
				{
					//
					//	walk from left
					//

					const hkpBpEndPoint* endP = axis.m_endPoints.begin() + (axis.m_endPoints.getSize()-4);

					for( p = &axis.m_endPoints[1];  p < endP && p[3].m_value <= aabbMin[i]; p += 4 )
					{
						bitField[ p[0].m_nodeIndex ] ^= mask;
						bitField[ p[1].m_nodeIndex ] ^= mask;
						bitField[ p[2].m_nodeIndex ] ^= mask;
						bitField[ p[3].m_nodeIndex ] ^= mask;
					}

					for ( ;p->m_value <= aabbMin[i]; p++ )
					{
						bitField[ p->m_nodeIndex ] ^= mask;
					}
					startEpMin[i] = p;

					for ( ;p->m_value <= aabbMax[i]; p++ )
					{
						bitField[ p->m_nodeIndex ] ^= mask & (p->isMaxPoint()-1); // flip bit, only if it is not a maxpoint
					}
					startEpMax[i] = p;
				}
				else
				{
					//
					//	walk from right
					//

					const hkpBpEndPoint* endP = axis.m_endPoints.begin() + 4;

					for( p = &axis.m_endPoints[axis.m_endPoints.getSize()-2]; p >= endP && p[-3].m_value > aabbMax[i]; p -= 4 )
					{
						bitField[ p[ 0].m_nodeIndex ] ^= mask;
						bitField[ p[-1].m_nodeIndex ] ^= mask;
						bitField[ p[-2].m_nodeIndex ] ^= mask;
						bitField[ p[-3].m_nodeIndex ] ^= mask;
					}

					for ( ;p->m_value > aabbMax[i]; p--)
					{
						bitField[ p->m_nodeIndex ] ^= mask;
					}
					startEpMax[i] = p+1;

					for ( ;p->m_value > aabbMin[i]; p--)
					{
						bitField[ p->m_nodeIndex ] ^= mask & - p->isMaxPoint();	// flip bits only for maxPoints
					}
					startEpMin[i] = p+1;
				}
#else
				const hkpBpEndPoint* midP = hkGetArrayElemUsingCache( axis.m_endPoints.begin(), axis.m_endPoints.getSize()>>1, g_SpuCollideUntypedCache, HK_SPU_UNTYPED_CACHE_LINE_SIZE );
				hkSpuReadOnlyIterator<hkp3AxisSweep::hkpBpEndPoint, 256, hkSpuWorldLinearCastDmaGroups::GET_BROADPHASE> p;

				if ( aabbMin[i] < midP->m_value )
				{
					const hkpBpEndPoint* endP = axis.m_endPoints.begin() + axis.m_endPoints.getSize();
					p.init( &axis.m_endPoints[1] );

					for ( ; p < endP &&  p->m_value <= aabbMin[ HK_HINT_SIZE16(i) ]; p++ )
					{
						bitField[ HK_HINT_SIZE16(p->m_nodeIndex) ] ^= mask;
					}
					startEpMin[i] = p;

					for ( ;p->m_value <= aabbMax[ HK_HINT_SIZE16(i) ]; p++ )
					{
						bitField[ HK_HINT_SIZE16(p->m_nodeIndex) ] ^= mask & (p->isMaxPoint()-1); // flip bit, only if it is not a maxpoint
					}
					startEpMax[i] = p;

				}
				else
				{
					const hkp3AxisSweep::hkpBpEndPoint* endP = axis.m_endPoints.begin();
					p.init ( &axis.m_endPoints[axis.m_endPoints.getSize()-2] );
					for ( ; p >= endP &&  p->m_value > aabbMax[ HK_HINT_SIZE16(i) ]; p-- )
					{
						bitField[ HK_HINT_SIZE16(p->m_nodeIndex) ] ^= mask;
					}
					startEpMax[i] = p+1;

					for ( ;p->m_value > aabbMin[ HK_HINT_SIZE16(i) ]; p--)
					{
						bitField[ HK_HINT_SIZE16(p->m_nodeIndex) ] ^= mask & -p->isMaxPoint();	// flip bits only for maxPoints
					}
					startEpMin[i] = p+1;
				}
#endif
				mask = mask << 1;
			}
		}
	}




	//
	//	Search for start overlaps
	//
	HK_INTERNAL_TIMER_SPLIT_LIST( "StartOverlaps");
	hkReal	minHitFraction = 1.f;		// Our early out variable
	{
		const hkpBpNode *node = &m_nodes[0];
		int numNodes = m_nodes.getSize();

		unsigned char* bitField = scratchArea.begin();
		const int* bitsInt = reinterpret_cast<const int *>( bitField );
		const int* end = bitsInt + (numNodes>>2) + 1;
		for( ; bitsInt < end; node += 4, bitsInt++ ) //check four at a time
		{
			if ( ((bitsInt[0] + 0x01010101 ) & 0x08080808) ==0 )
				continue;

			const unsigned char* bitChar = reinterpret_cast<const unsigned char*>(bitsInt);
#ifndef HK_PLATFORM_SPU
			if ( bitChar[0]==7 && !node[0].isMarker())
				minHitFraction = hkMath::min2(minHitFraction , collector.addBroadPhaseHandle(node[0].m_handle,0));
			if ( bitChar[1]==7 && !node[1].isMarker())
				minHitFraction = hkMath::min2(minHitFraction , collector.addBroadPhaseHandle(node[1].m_handle,0));
			if ( bitChar[2]==7 && !node[2].isMarker())
				minHitFraction = hkMath::min2(minHitFraction , collector.addBroadPhaseHandle(node[2].m_handle,0));
			if ( bitChar[3]==7 && !node[3].isMarker())
				minHitFraction = hkMath::min2(minHitFraction , collector.addBroadPhaseHandle(node[3].m_handle,0));
#else
			hkp3AxisSweep::hkpBpNode bpNode;
			bpNode = getNode( node, 0 );
			if ( bitChar[0]==7 && !bpNode.isMarker())
			{	minHitFraction = hkMath::min2(minHitFraction , collector.addBroadPhaseHandle(bpNode.m_handle,0)); }

			bpNode = getNode( node, 1 );
			if ( bitChar[1]==7 && !bpNode.isMarker())
			{	minHitFraction = hkMath::min2(minHitFraction , collector.addBroadPhaseHandle(bpNode.m_handle,0)); }

			bpNode = getNode( node, 2 );
			if ( bitChar[2]==7 && !bpNode.isMarker())
			{	minHitFraction = hkMath::min2(minHitFraction , collector.addBroadPhaseHandle(bpNode.m_handle,0)); }

			bpNode = getNode( node, 3 );
			if ( bitChar[3]==7 && !bpNode.isMarker())
			{	minHitFraction = hkMath::min2(minHitFraction , collector.addBroadPhaseHandle(bpNode.m_handle,0)); }
#endif
		}
	}

	HK_INTERNAL_TIMER_SPLIT_LIST( "Walk");

	//
	//	Initialize the invScale and offset in a way, so that the ray
	//  goes from 0,0,0 to 1,1,1. If the ray is flat in one direction
	//  than the start and end value is set to 2
	//  Also swap epMin and epMax, so that epMax always enters AABBs, and
	//  epMin always leaves AABBs
	//
	hkVector4 invScale;
	hkVector4 invOffsetMin;
	hkVector4 invOffsetMax;
	hkLong direction[3];	// set to +- sizeof(void*)
	int forwardFlags[3];	// set to 1 of dir is positive, set to 0 if direction is negative
	int reverseFlags[3];	// set to 0 of dir is positive, set to 1 if direction is negative
	{

		hkVector4 dir; dir.setSub( input.m_to, input.m_from );
		hkVector4 delta; delta.setMul(m_scale, dir);
		hkVector4 fabsDelta; fabsDelta.setAbs( delta );
		hkVector4 invDelta; invDelta.setReciprocal(delta);

		invScale = invDelta;
		hkVector4 i2fCorrection; i2fCorrection.setAll(m_intToFloatFloorCorrection);

		const hkVector4Comparison deltaGreaterZero = delta.greaterZero();
		hkVector4 swappedFromMin; swappedFromMin.setSelect(deltaGreaterZero, fromMin, fromMax);
		hkVector4 swappedFromMax; swappedFromMax.setSelect(deltaGreaterZero, fromMax, fromMin);

		hkVector4Comparison comp;
		{
			const hkSimdReal vEps = hkSimdReal::getConstant<HK_QUADREAL_EPS>();
			hkVector4 epsScale0; epsScale0.setAdd(swappedFromMin, m_offsetLow); epsScale0.mul(m_scale); epsScale0.mul(vEps);
			const hkVector4Comparison c0 = fabsDelta.less(epsScale0);
			hkVector4 epsScale1; epsScale1.setAdd(input.m_to, m_offsetLow); epsScale1.mul(m_scale); epsScale1.mul(vEps);
			const hkVector4Comparison c1 = fabsDelta.less(epsScale1);
			comp.setOr(c0,c1);
		}

		{
			invOffsetMin.setAdd(swappedFromMin, m_offsetLow); 
			invOffsetMin.mul(m_scale);
			invOffsetMin.sub(i2fCorrection);
			invOffsetMin.mul(invDelta);
		}
		{
			invOffsetMax.setAdd(swappedFromMax, m_offsetLow);
			invOffsetMax.mul(m_scale);
			invOffsetMax.sub(i2fCorrection);
			invOffsetMax.mul(invDelta);
		}

		for (int i=0; i<3; i++)
		{
			hkVector4ComparisonMask::Mask mask = hkVector4Comparison::getMaskForComponent(i);

			if ( deltaGreaterZero.anyIsSet(mask) )
			{
				direction[i] = hkSizeOf( hkpBpEndPoint );
				reverseFlags[i] = 0;
				forwardFlags[i] = 1;
			}
			else
			{
				direction[i] = - hkSizeOf( hkpBpEndPoint );
				hkAlgorithm::swap( startEpMin[i], startEpMax[i] );
				hkAlgorithm::swap( aabbMin[i], aabbMax[i] );
				startEpMin[i]--;	// if we walk left, the next boundary is left
				startEpMax[i]--;	// if we walk left, the next boundary is left
				reverseFlags[i] = 1;
				forwardFlags[i] = 0;
			}

			//
			//	Check for allowed division
			//
			if ( comp.anyIsSet(mask) )
			{
				invScale.zeroComponent(i);
				invOffsetMin(i) = -2.0f;
				invOffsetMax(i) = -2.0f;
			}
		}
	}

	//
	// test start point (see docu at start of this function
	//
#ifdef HK_DEBUG
	if (1)
	{
		for (int i = 0; i <3; i++ )
		{
			if ( invScale(i) == 0.0f ) { continue; }
			if ( aabbMin[i] == AABB_MIN_VALUE ) { continue; }
			if ( aabbMin[i] >= AABB_MAX_VALUE-1 ) { continue; }

			// test
			hkReal testMin =  aabbMin[i     ] * invScale(i) - invOffsetMin(i);
			hkReal testMax = (aabbMin[i] + 1) * invScale(i) - invOffsetMin(i);
			if ( direction[i] < 0 ) { testMin *= -1.0f;	testMax *= -1.0f; }
			hkReal numericalTolerance = hkMath::fabs( invOffsetMin(i) ) * HK_REAL_EPSILON * 16.0f;

			// Avoiding HK_ASSERT2 due to internal compiler errors (see JRA-1292)
			HK_ASSERT2(0x6815aa38,  testMin <= numericalTolerance && testMax >= -numericalTolerance, "Numerical inconsistency in hkp3AxisSweep");
		}
	}
#endif


	hkVector4 curDistanceMin;
	hkVector4 curDistanceMax;

#ifndef HK_PLATFORM_SPU
	const hkpBpEndPoint* epMin[3];
	const hkpBpEndPoint* epMax[3];
	epMin[0] = startEpMin[0];
	epMin[1] = startEpMin[1];
	epMin[2] = startEpMin[2];

	epMax[0] = startEpMax[0];
	epMax[1] = startEpMax[1];
	epMax[2] = startEpMax[2];
#else
	hkSpuReadOnlyIterator<hkp3AxisSweep::hkpBpEndPoint, 256,  hkSpuWorldRayCastDmaGroups::GET_BROADPHASE> epMin[3];
	hkSpuReadOnlyIterator<hkp3AxisSweep::hkpBpEndPoint, 256,  hkSpuWorldRayCastDmaGroups::GET_BROADPHASE> epMax[3];
	epMin[0].init( startEpMin[0] );
	epMin[1].init( startEpMin[1] );
	epMin[2].init( startEpMin[2] );

	epMax[0].init( startEpMax[0] );
	epMax[1].init( startEpMax[1] );
	epMax[2].init( startEpMax[2] );
#endif

	{
		hkVector4 epMinVals; epMinVals.set(epMin[0]->m_value, epMin[1]->m_value, epMin[2]->m_value);
		curDistanceMin.setMul(epMinVals, invScale);
		curDistanceMin.sub(invOffsetMin);
	}
	{
		hkVector4 epMaxVals; epMaxVals.set(epMax[0]->m_value, epMax[1]->m_value, epMax[2]->m_value);
		curDistanceMax.setMul(epMaxVals, invScale);
		curDistanceMax.sub(invOffsetMax);
	}

	unsigned char* bitField = scratchArea.begin();
	bitField[0] = 8;			// flag node 0

	hkReal dummy0;
	int componentMin = getMinComponent( curDistanceMin, dummy0 );
	int componentMax = getMinComponent( curDistanceMax, dummy0 );

	while(1) {
		if ( curDistanceMax(componentMax) < curDistanceMin(componentMin) )
		{
			//
			//	Check the max section
			//
			//	Early out if distance is too big
			if ( curDistanceMax(componentMax) > minHitFraction )
			{	break; 	}

			while(1)	// quick loop
			{
				{
					const int nodeindex = epMax[componentMax]->m_nodeIndex;
					int flip = BIT_SHIFT_LEFT(forwardFlags[componentMax] ^ epMax[componentMax]->isMaxPoint(), componentMax);
					bitField[nodeindex] ^= flip;

					if (bitField[nodeindex] >= 7)
					{
						if ( nodeindex == 0 )
						{	// just encountered the boundary object, so do not continue to walk in this direction
							curDistanceMax(componentMax) = 2;
							hkReal dummy; componentMax = getMinComponent( curDistanceMax, dummy );
							break; // quick loop
						}
						//entering an AABB
#ifndef HK_PLATFORM_SPU
						const hkpBpNode& node = m_nodes[nodeindex];
#else
						const hkpBpNode node = getNode( m_nodes.begin(), nodeindex);
#endif
						// Only add hit to collector if we have _just_ set the bit.
						// See HVK-3126 and HVK-3495
						if ( flip && !node.isMarker() )
						{
							minHitFraction = hkMath::min2(minHitFraction , collector.addBroadPhaseHandle(node.m_handle,0));
						}
					}
				}

				//
				//	walk: quick check, whether we really moved a distance forward, if no, quickloop
				//  else recalculate the distances and the min component
				//
				{
					BpInt lastValue = epMax[componentMax]->m_value;
#ifndef HK_PLATFORM_SPU
					epMax[componentMax] = hkAddByteOffsetConst<hkpBpEndPoint>(epMax[componentMax], direction[componentMax] );
#else
					if (direction[componentMax] > 0)
					{
						epMax[componentMax]++;
					}
					else
					{
						epMax[componentMax]--;
					}
#endif
					BpInt newValue = epMax[componentMax]->m_value;
					if ( lastValue == newValue )
					{
						continue; // quick loop
					}
					hkpBpEndPoint endPt = *epMax[componentMax]; // Need to make a copy if we're on the SPU
					calcCurDist( componentMax, curDistanceMax, &endPt, invScale, invOffsetMax );
					hkReal dummy; componentMax = getMinComponent( curDistanceMax, dummy );
					break;
				}
			}
		}
		else
		{
			//
			//	Check the min section
			//
			//	Early out if distance is too big
			if ( curDistanceMin(componentMin) > minHitFraction )
			{	break; 	}

			while(1)	// quick loop
			{
				const int nodeindex = epMin[componentMin]->m_nodeIndex;
				bitField[nodeindex] ^= BIT_SHIFT_LEFT(reverseFlags[componentMin] ^ epMin[componentMin]->isMaxPoint(), componentMin);

				if (bitField[nodeindex] > 8)
				{
					// just encountered the boundary object, so do not continue to walk in this direction
					HK_ASSERT(0x1a80a92f,  nodeindex == 0 );
					curDistanceMin(componentMin) = 2;
					hkReal dummy; componentMin = getMinComponent( curDistanceMin, dummy );
					break; // quick loop
				}

				//
				//	walk: quick check, whether we really moved a distance forward, if no, quickloop
				//  else recalculate the distances and the min component
				//
				{
					BpInt lastValue = epMin[componentMin]->m_value;
#ifndef HK_PLATFORM_SPU
					epMin[componentMin] = hkAddByteOffsetConst<hkpBpEndPoint>(epMin[componentMin], direction[componentMin] );
#else
					if (direction[componentMin] > 0)
					{
						epMin[componentMin]++;
					}
					else
					{
						epMin[componentMin]--;
					}
#endif
					BpInt newValue = epMin[componentMin]->m_value;
					if ( lastValue == newValue )
					{
						continue; // quick loop
					}

					hkpBpEndPoint endPt = *epMin[componentMin]; // Need to make a copy if we're on the SPU
					calcCurDist( componentMin, curDistanceMin, &endPt, invScale, invOffsetMin );
					hkReal dummy; componentMin = getMinComponent( curDistanceMin, dummy );

					break;
				}
			}

		}
	}
	HK_INTERNAL_TIMER_END_LIST();
}


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::getAabbFromNode(const hkpBpNode& node, hkAabb & aabb) const
{
	HK_ASSERT2(0xaf542fe4, m_scale.length<4>().getReal() != 0, "Make sure to call set32BitOffsetAndScale() after creating the broadphase.");

	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RO );

	hkVector4 mins; mins.set(m_axis[0].m_endPoints[node.min_x].m_value, m_axis[1].m_endPoints[node.min_y].m_value, m_axis[2].m_endPoints[node.min_z].m_value);
	hkVector4 maxs; maxs.set(m_axis[0].m_endPoints[node.max_x].m_value, m_axis[1].m_endPoints[node.max_y].m_value, m_axis[2].m_endPoints[node.max_z].m_value);

	hkVector4 invScale; invScale.setReciprocal(m_scale);

	hkVector4 minV;
	minV.setMul(mins, invScale);
	minV.sub( m_offsetLow );

	hkVector4 maxV;
	maxV.setMul(maxs, invScale);
	maxV.sub( m_offsetLow );

	aabb.m_min = minV;
	aabb.m_max = maxV;
}
#endif


#if !defined(HK_PLATFORM_SPU)
//- If you set the broadphase size to be 32768 you'll get a range
//  of -16384 to 16384 which will be remapped to
//  0 to 65536-4 using even numbers only.
//
//- So if you want to shift by a full meter always so that
//  effectiveShiftDistance always matches your shift,
//  your broadphase size must confirm to the following rule:
//          0xfffc/size = integer
//  or      size = integerValue/0xfffc
//  However your effective shiftDistance might always be off by
//  exactly 1.0. In this due to the fact that we are using a
//  fast floating point to int conversion. This behavior can
//  be different on different platforms. Simply correct your
//  input shift distance by 1.0 in this case.

void hkp3AxisSweep::shiftAllObjects( const hkVector4& shiftDistance, hkVector4& effectiveShiftDistanceOut, hkArray<hkpBroadPhaseHandlePair>& newCollisionPairs )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );
	//checkConsistency();

	HK_ASSERT2( 0xf0160403, getNumMarkers()==0, "You cannot use this function when markers are enabled" );
	HK_ASSERT2(0xaf542fe5, m_scale.length<4>().getReal() != 0, "Make sure to call set32BitOffsetAndScale() after creating the broadphase.");

	//
	// first try to make a good guess for the real distance we have to shift
	//
	hkVector4 realShift = shiftDistance;
	hkInt64   iShift[3];
	{
		for (int i = 0; i < 3; i++)
		{
			iShift[i] = hkClearBits(hkInt64( realShift(i) * m_scale(i) ), 1);
			realShift(i) = iShift[i] / m_scale(i);
		}
		effectiveShiftDistanceOut = realShift;
	}

	hkArray<hkpBroadPhaseHandlePair> addPairs;
	hkArray<hkpBroadPhaseHandlePair> deletedPairsDummy;

	//
	//	shift all objects except objects sitting already at the border
	//
	{
		for (int a = 0; a < 3; a++)
		{
			hkpBpAxis& axis = m_axis[a];
			int shift = iShift[a];

			int start, end, step;
			
			if ( shift < 0 )
			{
				start = 1;
				end   = axis.m_endPoints.getSize()-1;
				step  = 1;
			}
			else
			{
				start = axis.m_endPoints.getSize()-2;
				end   = 0;
				step  = -1;
			}

			for ( int i = start; i != end; i+=step )
			{
				hkpBpEndPoint& ep = axis.m_endPoints[i];
#ifdef HK_BROADPHASE_32BIT
				hkInt64 val;
				hkInt64 newVal;
#else
				int val;
				int newVal;
#endif
				val = ep.m_value;

				// Don't shift borders. They are snapped to [0, 1] or [AABB_MAX_FVALUE, AABB_MAX_FVALUE+1] in addObject.
				if ( val <= 1 || val >= AABB_MAX_FVALUE )
				{
					continue;
				}

				newVal = ((val + shift) & ~1) | (val&1);	// preserve last bit = max point flag

				hkpBpNode& node = m_nodes[ep.m_nodeIndex];

				// clip values
				if ( newVal < 0 )
				{
					newVal = val & 1;	// preserve its max point
				}
				else if ( newVal >= AABB_MAX_FVALUE )
				{
					newVal = AABB_MAX_FVALUE | (val & 1);
				}
				ep.m_value = newVal; 


				if ( newVal == 0 || newVal == (AABB_MAX_FVALUE | 1) )
				{
					hkAabbUint32 newAabb;
					newAabb.m_min[0] = m_axis[0].m_endPoints[ node._getMin<0>() ].m_value;
					newAabb.m_min[1] = m_axis[1].m_endPoints[ node._getMin<1>() ].m_value;
					newAabb.m_min[2] = m_axis[2].m_endPoints[ node._getMin<2>() ].m_value;
					newAabb.m_max[0] = m_axis[0].m_endPoints[ node._getMax<0>() ].m_value;
					newAabb.m_max[1] = m_axis[1].m_endPoints[ node._getMax<1>() ].m_value;
					newAabb.m_max[2] = m_axis[2].m_endPoints[ node._getMax<2>() ].m_value;

					updateAabb( node.m_handle, newAabb, newCollisionPairs, deletedPairsDummy );
					HK_ASSERT2( 0xf0234354, deletedPairsDummy.isEmpty(), "Found deleted pairs in shift broadphase." );
				}
			}
		}
	}
	CHECK_CONSISTENCY();
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkp3AxisSweep::shiftBroadPhase( const hkVector4& shiftDistance, hkVector4& effectiveShiftDistanceOut, hkArray<hkpBroadPhaseHandlePair>& newCollisionPairs )
{
	HK_ACCESS_CHECK_OBJECT( this, HK_ACCESS_RW );

	HK_ASSERT2(0xaf542fe6, m_scale.length<4>().getReal() != 0, "Make sure to call set32BitOffsetAndScale() after creating the broadphase.");

	hkVector4 shiftObjects; shiftObjects.setNeg<4>( shiftDistance );
	hkVector4 effectiveObjectShift;

	shiftAllObjects( shiftObjects, effectiveObjectShift, newCollisionPairs );

	this->m_offsetLow.add( effectiveObjectShift );
	hkVector4 rounding; 
	rounding.setReciprocal( m_scale );
	rounding.setComponent<3>(hkSimdReal_1);
	m_offsetHigh.setAdd( m_offsetLow, rounding );

	this->m_offsetLow32bit.add( effectiveObjectShift );
	hkVector4 rounding32bit; 
	rounding32bit.setReciprocal( m_scale32bit );
	rounding32bit.setComponent<3>(hkSimdReal_1);
	m_offsetHigh32bit.setAdd( m_offsetLow32bit, rounding32bit );

	effectiveShiftDistanceOut.setNeg<4>( effectiveObjectShift );
}
#endif


#if !defined(HK_PLATFORM_SPU)
const hkAabb& hkp3AxisSweep::getOriginalAabb() const
{
	return m_aabb;
}
#endif

void hkp3AxisSweep::checkDeterminism()
{
#if defined HK_ENABLE_DETERMINISM_CHECKS
	HK_TIME_CODE_BLOCK("hkp3AxisSweep::checkDeterminism", HK_NULL );

	for (int ai = 0; ai < 3; ai++)
	{
		hkpBpAxis& axis = m_axis[ai];
		hkCheckDeterminismUtil::checkMtCrc(  0xf00001e0, axis.m_endPoints.begin(), axis.m_endPoints.getSize() );
	}
	for (int ni = 0; ni < m_nodes.getSize(); ni++)
	{
		hkpBpNode& node = m_nodes[ni];
		if (node.m_handle	&& !node.isMarker() ) // we flag markers this way!
		{
			hkpLinkedCollidable* coll = static_cast<hkpLinkedCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(node.m_handle)->getOwner() );
			hkpRigidBody* rb = hkpGetRigidBody( coll) ;
			if (rb)
			{
				hkCheckDeterminismUtil::checkMt( 0xf00001e9, rb->getUid());
			}
		}
	}
#endif
}

//
// Code stripping support
//

#if !defined(HK_PLATFORM_SPU)
#	ifndef HK_BROADPHASE_32BIT

hkpBroadPhase* HK_CALL hk3AxisSweep16CreateBroadPhase( const hkVector4& worldMin, const hkVector4& worldMax, int numMarkers )
{
	HK_OPTIONAL_COMPONENT_MARK_USED(hkp3AxisSweep);
	return new hkp3AxisSweep( worldMin, worldMax, numMarkers );
}
HK_OPTIONAL_COMPONENT_DEFINE(hkp3AxisSweep, hkpBroadPhase::s_createSweepAndPruneBroadPhaseFunction, hk3AxisSweep16CreateBroadPhase);


#	endif
#endif

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
