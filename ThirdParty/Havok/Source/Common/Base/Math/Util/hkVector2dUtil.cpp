/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Util/hkVector2dUtil.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Math/Vector/hkVector2d.h>

namespace hkVector2Util
{

namespace
{
 	struct IndexedLess
 	{
		IndexedLess( const hkArrayBase<hkVector2d>& a ) : m_a(a) {  }
		hkBool32 operator()( int i, int j )
		{
			const hkVector2d& pi = m_a[i];
			const hkVector2d& pj = m_a[j];
			return pi.x < pj.x || (pi.x == pj.x && pi.y < pj.y);
		}
		const hkArrayBase<hkVector2d>& m_a;
	};

	template< typename T >
	struct Deque
	{
			// invariant: m_bot is the index of the first valid item
			// m_top index of one past the end
			// m_top-m_bot == number of elements
		Deque(int n) : m_vals(n), m_bot(1 + n/2), m_top(m_bot)
		{
			HK_ON_DEBUG( hkString::memSet(m_vals.begin(), 0xff, n*sizeof(T)) );
		}

		void pushBot( T t )
		{
			m_vals[--m_bot] = t;
		}
		void pushTop( T t )
		{
			m_vals[m_top++] = t;
		}
		void popBot()
		{
			HK_ON_DEBUG( m_vals[m_bot] = -1 );
			++m_bot;
			HK_ASSERT(0x4d4c4a63, m_bot <= m_top );
		}
		void popTop()
		{
			HK_ON_DEBUG( m_vals[m_top-1] = -1 );
			--m_top;
			HK_ASSERT(0x4d4c4a63, m_bot <= m_top );
		}
		T bot(int i)
		{
			return m_vals[m_bot+i];
		}
		T top(int i)
		{
			return m_vals[m_top-1-i];
		}
		void get(hkArray<T>& out)
		{
			int n = m_top - m_bot;
			hkArray<T> active( m_vals.begin()+m_bot, n, n ); // noncopying ctor
			out = active;
		}

		hkLocalBuffer<T> m_vals;
		int m_bot;
		int m_top;
	};
}


void HK_CALL convexHullSimplePolyline( const hkArrayBase<hkVector2d>& line, hkArray<int>& indicesOut )
{
	const int NPOINT = line.getSize();
	Deque<int> deque( 2 * (NPOINT-1) );

	if( line[2].leftOfLine(line[0], line[1]) )
	{
		deque.pushBot( 0 );
		deque.pushTop( 1 );
	}
	else
	{
		deque.pushBot( 1 );
		deque.pushTop( 0 );
	}
	deque.pushBot( 2 );
	deque.pushTop( 2 );

	hkVector2d bot0 = line[ deque.bot(0) ];
	hkVector2d bot1 = line[ deque.bot(1) ];
	hkVector2d top0 = line[ deque.top(0) ];
	hkVector2d top1 = line[ deque.top(1) ];

 	for( int i = 3; i < NPOINT; ++i )
	{
		if( line[i].leftOfLine(bot0, bot1) && line[i].leftOfLine(top1, top0) )
		{
			continue;
		}
		while( ! line[i].leftOfLine(bot0, bot1) )
		{
			deque.popBot();
			bot0 = bot1;
			bot1 = line[ deque.bot(1) ];
		}
		deque.pushBot(i);
		bot1 = bot0;
		bot0 = line[deque.bot(0) ];

		while( ! line[i].leftOfLine(top1, top0) )
		{
			deque.popTop();
			top0 = top1;
			top1 = line[ deque.top(1) ];
		}
		deque.pushTop(i);
		top1 = top0;
		top0 = line[deque.top(0) ];
  	}

	deque.get( indicesOut );
}

// http://geometryalgorithms.com/Archive/algorithm_0109/algorithm_0109.htm#Monotone%20Chain
hkResult HK_CALL convexHullIndices( const hkArrayBase<hkVector2d>& originalPoints, hkArrayBase<int>& indicesOut )
{
	
	// Can we save mem here: Overwrite 'orignalPoints' array
	int NPOINT = originalPoints.getSize();
	hkLocalBuffer<hkVector2d> points( NPOINT );
	hkLocalBuffer<int> originalIndices( NPOINT );
	if (points.begin() == HK_NULL || originalIndices.begin() == HK_NULL)
	{
		return HK_FAILURE;
	}

	{
		for( int i = 0; i < NPOINT; ++i )
		{
			originalIndices[i] = i;
		}
		hkSort( originalIndices.begin(), NPOINT, IndexedLess(originalPoints) );
		int di = 0;
		points[0] = originalPoints[originalIndices[0]];
		for( int si = 1; si < NPOINT; ++si )
		{
			const hkVector2d& p = originalPoints[ originalIndices[si] ];
			if( points[di].x != p.x || points[di].y != p.y )
			{
				di += 1; // remove duplicates
				points[di] = p;
				originalIndices[di] = originalIndices[si];
			}
		}
		NPOINT = di + 1;
		//indicesOut.reserve( NPOINT+1 );
		HK_ASSERT2(0x52c91212, indicesOut.getCapacity() >= NPOINT+1, "You must set the capacity of indicesOut before calling this method." );
	}

	
	// Can we get rid of this array?
	hkLocalArray<hkVector2d> pointsOut( NPOINT );
	if (pointsOut.begin() == HK_NULL)
	{
		return HK_FAILURE;
	}
		
	int minXminY = 0;
	int minXmaxY;
	int maxXminY;
	int maxXmaxY = NPOINT-1;

	// get start and end walking indices
	{
		{
			hkDouble64 minXpos = points[0].x;
			for( minXmaxY = 1; minXmaxY < NPOINT && points[minXmaxY].x == minXpos; ++minXmaxY )
			{
			}
		}
		if( minXmaxY == NPOINT ) // degenerate - all x == xmin
		{
			indicesOut.pushBackUnchecked( 0 );
			if( points[NPOINT-1].y != points[0].y ) // segment or simple point?
			{
				indicesOut.pushBackUnchecked( NPOINT-1 );
				indicesOut.pushBackUnchecked( 0 );
			}
			return HK_SUCCESS;
		}
		minXmaxY -= 1;
		{
			hkDouble64 maxXpos = points[NPOINT-1].x;
			int i;
			for( i = NPOINT-2; i >= 0 && points[i].x == maxXpos; --i )
			{
			}
			maxXminY = i + 1;
		}
	}

	// walk forwards along along bottom
	{
		hkVector2d minXminYpoint = points[minXminY];
		hkVector2d maxXminYpoint = points[maxXminY];

		indicesOut.pushBackUnchecked( originalIndices[minXminY] );
		pointsOut.pushBackUnchecked( minXminYpoint );

		for( int i = minXmaxY; i <= maxXminY; ++i )
		{
			hkVector2d p = points[i];
			if( ! p.rightOfLine( minXminYpoint, maxXminYpoint) && i < maxXminY )
			{
				continue;
			}
			while( pointsOut.getSize() >= 2 )
			{
				int n = pointsOut.getSize();
				if( p.leftOfLine( pointsOut[n-2], pointsOut[n-1] ) )
				{
					break;
				}
				indicesOut.popBack();
				pointsOut.popBack();
			}
			indicesOut.pushBackUnchecked( originalIndices[i] );
			pointsOut.pushBackUnchecked( p );
		}
	}

	// walk backwards along top.
	{
		pointsOut.clear();
		if( maxXmaxY != maxXminY )
		{
			indicesOut.pushBackUnchecked( originalIndices[maxXmaxY] );
		}
		pointsOut.pushBackUnchecked( points[NPOINT-1] );

		hkVector2d maxXmaxYpoint = points[NPOINT-1];
		hkVector2d minXmaxYpoint = points[minXmaxY];

		for( int i = maxXminY-1; i >= minXmaxY; --i )
		{
			hkVector2d p = points[i];
			if( ! p.rightOfLine( maxXmaxYpoint, minXmaxYpoint) && i > minXmaxY )
			{
				continue;
			}
			while( pointsOut.getSize() >= 2 )
			{
				int n = pointsOut.getSize();
				if( p.leftOfLine( pointsOut[n-2], pointsOut[n-1] ) )
				{
					break;
				}
				indicesOut.popBack();
				pointsOut.popBack();
			}
			indicesOut.pushBackUnchecked( originalIndices[i] );
			pointsOut.pushBackUnchecked( p );
		}

		if( minXmaxY != minXminY )
		{
			indicesOut.pushBackUnchecked( originalIndices[minXminY] );
		}
	}
	//for( int i = 1; i < indicesOut.getSize(); ++i )	{ HK_ASSERT(0x5c35248e, indicesOut[i] != indicesOut[i-1] ); }
	return HK_SUCCESS;
}
void HK_CALL convexHullVertices( const hkArrayBase<hkVector2d>& points, hkArray<hkVector2d>& hullOut )
{
	HK_ASSERT(0x1a776041, points.getSize() );
	hkArray<int>::Temp indices; indices.setSize( points.getSize() );
	convexHullIndices( points, indices );
	hullOut.reserveExactly( indices.getSize() );
	for( int i = 0; i < indices.getSize(); ++i )
	{
		hullOut.pushBackUnchecked( points[indices[i]] );
	}
}

hkBool HK_CALL edgesIntersect( const hkVector2d& a, const hkVector2d& b, const hkVector2d& c, const hkVector2d& d )
{
	// Solving for a + r(b-a) = c + s(d-c)
	hkDouble64 denom = (b.x - a.x) * (d.y - c.y) - (b.y - a.y) * (d.x - c.x);
	hkDouble64 rNum = (a.y - c.y) * (d.x - c.x) - (a.x - c.x) * (d.y - c.y);
	hkDouble64 sNum = (a.y - c.y) * (b.x - a.x) - (a.x - c.x) * (b.y - a.y);

	// I don't need to worry about a divide by 0, because
	// if denom is 0, the equality below requires rNum > 0.0f && rNum < 0.0f... which can never happen

	/* If 0<=r<=1 & 0<=s<=1, intersection exists */
	if (denom < 0.0f)
	{
		denom = -denom;
		rNum = -rNum;
		sNum = -sNum;
	}

	return (rNum > 0.0f && sNum > 0.0f && rNum < denom && sNum < denom);

}


hkDouble64 computeObb(const hkArrayBase<hkVector2d>& points, hkVector2d& centerOut, hkVector2d& e0Out, hkVector2d& e1Out, hkResult& resOut)
{
	hkArray<int> hullIndices;
	hkResult hullIndicesRes = hullIndices.reserve(points.getSize()+1);
	if(hullIndicesRes != HK_SUCCESS)
	{
		resOut = HK_FAILURE;
		return 0.0f;
	}
	
	hkResult cvxHullRes = convexHullIndices(points, hullIndices);
	if(cvxHullRes != HK_SUCCESS)
	{
		resOut = HK_FAILURE;
		return 0.0f;
	}

	// convexHullIndices duplicates the first and last vertices. Remove the last one to avoid extra work.
	HK_ASSERT(0x3984fef6, hullIndices[0] == hullIndices.back() );
	hullIndices.popBack();

	hkDouble64 minArea = HK_DOUBLE_MAX;

	for (int i = 0, j = hullIndices.getSize()-1; i < hullIndices.getSize(); j = i, i++)
	{
		hkVector2d e0; e0.setSub(points[ hullIndices[i] ], points[ hullIndices[j] ]);
		if (e0.normalizeWithLength() == 0.f)
			continue;

		hkVector2d e1; e1.setPerp(e0);

		hkDouble64 min0 = 0.f, min1 = 0.f, max0 = 0.f, max1 = 0.f;
		for (int k = 0; k < hullIndices.getSize(); k++)
		{
			hkVector2d d; d.setSub(points[ hullIndices[k] ], points[ hullIndices[j] ]);
			hkDouble64 dot = d.dot(e0);
			if (dot < min0) min0 = dot;
			if (dot > max0) max0 = dot;
			dot = d.dot(e1);
			if (dot < min1) min1 = dot;
			if (dot > max1) max1 = dot;
		}
		hkDouble64 area = (max0 - min0) * (max1 - min1);

		if (area < minArea)
		{
			minArea = area;
			hkVector2d v;
			v.x = (min0+max0)*e0.x + (min1+max1)*e1.x;
			v.y = (min0+max0)*e0.y + (min1+max1)*e1.y;
			centerOut.setAddMul(points[ hullIndices[j] ], v, 0.5f);
			e0Out.setMul(e0, (max0-min0)*0.5f);
			e1Out.setMul(e1, (max1-min1)*0.5f);
		}
	}

	resOut = HK_SUCCESS;
	return minArea;
}

#ifdef HK_REAL_IS_DOUBLE
/// Helper to get the number of bytes allocated on stack for hull generation
int HK_CALL getStackSizeInBytesRequiredForConvexHullIndices(int inputPointsSize)
{
	return inputPointsSize * 
		( sizeof(hkVector2d)	+ //hkLocalBuffer<hkVector2d> points( NPOINT );
		 sizeof(int)	+ // hkLocalBuffer<int> originalIndices( NPOINT );
		 sizeof(hkVector2d)	// hkLocalArray<hkVector2d> pointsOut( NPOINT );
	);
}
#endif

} // namespace hkVector2Util

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
