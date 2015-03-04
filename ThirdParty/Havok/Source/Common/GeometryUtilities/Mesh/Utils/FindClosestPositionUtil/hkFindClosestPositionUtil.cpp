/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Utils/FindClosestPositionUtil/hkFindClosestPositionUtil.h>

#ifdef HK_DEBUG
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#endif

#if defined(HK_REAL_IS_DOUBLE) || defined(HK_COMPILER_HAS_INTRINSICS_IA32)
#include <Common/Base/Math/Vector/hkIntVector.h>
#define HACKY_INT_TO_FLOAT 0
#else
#define HACKY_INT_TO_FLOAT 1
#endif

hkFindClosestPositionUtil::hkFindClosestPositionUtil():
	m_boxFreeList(sizeof(Box), sizeof(void*), 4096)
{
}

HK_FORCE_INLINE /* static */int hkFindClosestPositionUtil_toIntFast23(hkFloat32 a)
{
	HK_ASSERT(0x43534534, a >= 0.0f);

	a += (8388608.0f);
	return int((*(hkInt32*)&a) & 0x7fffff);
}

void hkFindClosestPositionUtil::_calculateIntCoord(const hkVector4& point, IntCoord& coord) const
{
	HK_ASSERT(0x5345345a, m_aabb.containsPoint(point));

	hkVector4 p;
	p.setAdd(point, m_offset);
	p.mul(m_scale);

#if (HACKY_INT_TO_FLOAT == 0)
	HK_ASSERT(0x43534534, p.greaterEqualZero().allAreSet<hkVector4Comparison::MASK_XYZ>());
	hkIntVector p_i; 
	p_i.setConvertF32toS32(p);
	p_i.store<3,HK_IO_NATIVE_ALIGNED>((hkUint32*)&coord.m_x);
#else
	coord.m_x = hkFindClosestPositionUtil_toIntFast23(p(0));
	coord.m_y = hkFindClosestPositionUtil_toIntFast23(p(1));
	coord.m_z = hkFindClosestPositionUtil_toIntFast23(p(2));
#endif
}


void hkFindClosestPositionUtil::start(const hkAabb& aabbIn, hkReal threshold)
{
	// Save the threshold
	m_threshold.setFromFloat(threshold);

	hkSimdReal threshold4 = m_threshold;

	hkAabb aabb = aabbIn;

	// We want bigger so that generally the intersections will be totally inside a grid entry
	threshold4.mul(hkSimdReal_3);

	// If the threshold is too small it won't work, so we need to find the max direction and see if it will
	hkSimdReal maxExtent;
	{
		hkVector4 diff;
		diff.setSub(aabbIn.m_max, aabbIn.m_min);

		maxExtent = diff.horizontalMax<3>();

#if (HACKY_INT_TO_FLOAT == 1)
		// We are using a hacky way to get an int from a float - lets not make the range too large
		const hkSimdReal maxInt = hkSimdReal::fromInt32(0x3ffff);
		if (threshold4.getReal() < 1e-10f || (maxExtent / threshold4) > maxInt)
		{
			threshold4 = maxExtent / maxInt;
		}
#endif
	}

	aabb.m_min.setSub(aabb.m_min,threshold4);
	aabb.m_max.setAdd(aabb.m_max,threshold4);

	m_aabb = aabb;

	// Work out the scale and offset
	m_offset.setNeg<4>(aabb.m_min);

	{
		// Put in the appropriate range
		hkVector4 diff; diff.setSub(aabb.m_max, aabb.m_min);
		m_scale.setReciprocal(diff); 
		m_scale.setComponent<3>(hkSimdReal_1);
		hkSimdReal v; v.setDiv(maxExtent, threshold4);
		m_scale.mul(v);
	}

	m_positions.clear();
	m_boxFreeList.freeAllMemory();
	m_hashMap.clear();
}

void hkFindClosestPositionUtil::end()
{
	m_positions.clear();
	m_boxFreeList.freeAllMemory();
	m_hashMap.clear();
}

void hkFindClosestPositionUtil::_findClosest(const IntCoord& coord, const hkVector4& point, hkSimdReal& closestDistInOut, int& closestIndexInOut) const
{
	hkSimdReal closestDist = closestDistInOut;
	int closestIndex = closestIndexInOut;

	hkUint32 hash = coord.calculateHash();

	HashMap::Iterator iter = m_hashMap.findKey(hash);
	if (!m_hashMap.isValid(iter))
	{
		return;
	}

	{
		// See if we can add the point to a box that already exists
		Box* cur = m_hashMap.getValue(iter);
		for (; cur; cur = cur->m_next)
		{
			if (cur->m_coord == coord)
			{
				for (int i = 0; i < cur->m_numIndices; i++)
				{
					hkVector4 diff;

					const int index = cur->m_indices[i];
					diff.setSub(m_positions[index], point);

					hkSimdReal dist = diff.lengthSquared<3>();

					if (dist < closestDist)
					{
						closestIndex = index;
						closestDist = dist;

						if (dist.isLessEqualZero())
						{
							// If its exact then we are done
							break;
						}
					}
				}
			}
		}
	}

	closestDistInOut = closestDist;
	closestIndexInOut = closestIndex;
}

int hkFindClosestPositionUtil::findClosest(const hkVector4& point) const
{
	hkVector4 p;

	IntCoord min, max;
	p.setSub(point, m_threshold);
	_calculateIntCoord(p, min);

	p.setAdd(point, m_threshold);
	_calculateIntCoord(p, max);

	int closestIndex = -1;
	hkSimdReal closestDist = m_threshold * m_threshold;

	if (min == max)
	{
		// There is only one to test
		_findClosest(min, point, closestDist, closestIndex);
	}
	else
	{
		IntCoord coord;

		for (int x = min.m_x; x <= max.m_x; x++)
		{
			for (int y = min.m_y; y <= max.m_y; y++)
			{
				for (int z = min.m_z; z <= max.m_z; z++)
				{
					coord.m_x = x;
					coord.m_y = y;
					coord.m_z = z;

					// Find the closest
					_findClosest(coord, point, closestDist, closestIndex);
					if (closestDist.isLessEqualZero())
					{
						/// Its on top of it
						return closestIndex;
					}
				}
			}
		}
	}

	if (closestDist < m_threshold * m_threshold)
	{
		return closestIndex;
	}
	else
	{
		return -1;
	}
}

int hkFindClosestPositionUtil::addPoint(const hkVector4& point, hkResult* resultOut )
{
	hkResult res = HK_SUCCESS;
	IntCoord coord;
	_calculateIntCoord(point, coord);

	hkUint32 hash = coord.calculateHash();
	HashMap::Iterator iter = m_hashMap.findKey(hash);

	int index = m_positions.getSize();
	m_positions.pushBack(point);

	if (m_hashMap.isValid(iter))
	{
		// See if we can add the point to a box that already exists
		Box* first = m_hashMap.getValue(iter);
		Box* cur = first;
		for (; cur; cur = cur->m_next)
		{
			if (cur->m_coord == coord)
			{
				if (cur->m_numIndices < Box::MAX_INDICES)
				{
					// Found a box with space, and the index
					cur->m_indices[cur->m_numIndices++] = index;
					if (resultOut)
					{
						*resultOut = res;
					}
					return index;
				}
			}
		}

		// If we make it here, there was either no space, or no box has the same hash
		Box* box = (Box*)m_boxFreeList.alloc();

		if(box) // The allocation can potentially fail if memory is low
		{
			box->m_coord = coord;
			box->m_indices[0] = index;
			box->m_numIndices = 1;

			// Link in
			box->m_next = first->m_next;
			first->m_next = box;
		}
		else
		{
			res = HK_FAILURE;
		}
	}
	else
	{
		Box* box = (Box*)m_boxFreeList.alloc();

		if (box) // The allocation can potentially fail if memory is low
		{
			box->m_coord = coord;
			box->m_indices[0] = index;
			box->m_numIndices = 1;
			box->m_next = HK_NULL;

			// Add the box
			m_hashMap.tryInsert(hash, box, res);
		}
		else
		{
			res = HK_FAILURE;
		}
	}

	if (resultOut)
	{
		*resultOut = res;
	}

	return index;
}

hkResult hkFindClosestPositionUtil::_findEntry(hkVector4Parameter point, Box*& boxOut, int& indexOut, int& boxIndexOut)
{
	IntCoord coord;
	_calculateIntCoord(point, coord);

	hkUint32 hash = coord.calculateHash();
	HashMap::Iterator iter = m_hashMap.findKey(hash);

	if (m_hashMap.isValid(iter))
	{
		Box* first = m_hashMap.getValue(iter);
		Box* cur = first;
		for (; cur; cur = cur->m_next)
		{
			if (cur->m_coord == coord)
			{
				// Check each index in the box
				for (int i=0; i<cur->m_numIndices; i++)
				{
					hkVector4 p2 = m_positions[ cur->m_indices[i] ];
					if( point.distanceToSquared(p2).isLess(hkSimdReal_Eps) )
					{
						indexOut = cur->m_indices[i];
						boxOut = cur;
						boxIndexOut = i;
						return HK_SUCCESS;
					}
				}
			}
		}
	}

	return HK_FAILURE;
}
void hkFindClosestPositionUtil::removePoint( hkVector4Parameter point )
{
	Box* box = HK_NULL;
	int indexToRemove = -1, boxIndex = -1;

	hkResult foundPoint = _findEntry(point, box, indexToRemove, boxIndex);

	if (foundPoint == HK_SUCCESS)
	{
		box->m_indices[boxIndex] = box->m_indices[box->m_numIndices - 1];
		box->m_numIndices--;

		// We need to swap the point that we're removing with the last point in the position index
		if (m_positions.getSize()-1 != indexToRemove)
		{
			hkVector4 lastPos = m_positions.back();
			Box* lastBox = HK_NULL;
			int lastIndex = -1, lastBoxIndex = -1;

			HK_ON_DEBUG(hkResult foundLastPoint = ) _findEntry(lastPos, lastBox, lastIndex, lastBoxIndex);

			HK_ASSERT(0x64b5733e, foundLastPoint == HK_SUCCESS);
			// If the same exact point is added twice, this *might* fail
			// This isn't handled now, but we might be able to do it by searching ALL the boxes
			HK_ASSERT(0x7f3f330c, lastIndex == m_positions.getSize()-1);

			lastBox->m_indices[lastBoxIndex] = indexToRemove;
		}

		m_positions.removeAt(indexToRemove);
	}

}

void hkFindClosestPositionUtil::addPoints(const hkVector4* points, int numPoints)
{
	const hkVector4* _end = points + numPoints;
	for (; points != _end; points++)
	{
		addPoint(*points);
	}
}


int hkFindClosestPositionUtil::findClosestLinearly(const hkVector4& p) const
{
	hkSimdReal closestDist = hkSimdReal_Max;
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkIntVector minusOne; minusOne.splatImmediate32<-1>();
	hkIntVector closestIndex = minusOne;
	hkIntVector counter; counter.setZero();
	const hkIntVector one = hkIntVector::getConstant<HK_QUADINT_1>();
#else
	int closestIndex = -1;
#endif
	{
		const hkVector4* cur = m_positions.begin();
		const hkVector4* _end = m_positions.end();

		for (; cur != _end; cur++)
		{
			hkVector4 diff;
			diff.setSub(*cur, p);

			hkSimdReal dist = diff.lengthSquared<3>();

#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
			hkVector4Comparison lt = dist.less(closestDist);
			{
				closestDist.setSelect(lt, dist, closestDist);
				closestIndex.setSelect(lt, counter, closestIndex);
			}
			counter.setAddS32(counter, one);
#else
			if (dist < closestDist)
			{
				closestDist = dist;
				closestIndex = int(cur - m_positions.begin());
			}
#endif
		}
	}

#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4Comparison thr = closestDist.less(m_threshold * m_threshold);
	{
		closestIndex.setSelect(thr, closestIndex, minusOne);
		return closestIndex.getU32<0>();
	}
#else
	if (closestDist < m_threshold * m_threshold)
	{
		return closestIndex;
	}
	return -1;
#endif
}



#ifdef HK_DEBUG

/* static */hkResult HK_CALL hkFindClosestPositionUtil::selfCheck()
{
	hkFindClosestPositionUtil positionUtil;

	hkArray<hkVector4> positions;
	const int numPoints = 1000;
	hkPseudoRandomGenerator rand(100);

	hkSimdReal fifty; fifty.setFromFloat(50.0f);
	for (int i = 0; i < numPoints; i++)
	{
		hkVector4 v;
		rand.getRandomVector11(v);

		v.mul(fifty);
		positions.pushBack(v);
	}

	hkAabb aabb;
	aabb.m_max.setAll(fifty);
	aabb.m_min.setAll(-fifty);

	hkReal thresholds[] = { 1e-10f, 1e-5f, 1e-2f, 1.0f, 15.0f, 100.0f };
	for (unsigned int i = 0; i < HK_COUNT_OF(thresholds); i++)
	{
		const hkReal threshold = thresholds[i];

		positionUtil.start(aabb, threshold);

		// Add all the positions
		positionUtil.addPoints(positions.begin(), positions.getSize());

		for (int j = 0; j < positions.getSize() + 1000; j++)
		{
			hkVector4 v;

			if (j < positions.getSize())
			{
				v = positions[j];
			}
			else
			{
				rand.getRandomVector11(v);
				v.mul(fifty);
			}

			// Find the closest point
			int index = positionUtil.findClosest(v);
			int index2 = positionUtil.findClosestLinearly(v);

			if (index != index2)
			{
				if (index < 0 || index2 < 0)
				{
					//int check = positionUtil.findClosest(v);
					return HK_FAILURE;
				}

				hkVector4 diff;
				diff.setSub(positionUtil.getPoint(index), v);
				const hkSimdReal dist = diff.lengthSquared<3>();

				diff.setSub(positionUtil.getPoint(index2), v);
				const hkSimdReal dist2 = diff.lengthSquared<3>();

				if (dist.isNotEqual(dist2))
				{
					return HK_FAILURE;
				}
			}
		}
		positionUtil.end();
	}

	return HK_SUCCESS;
}

#endif

int HK_CALL hkFindClosestPositionUtil::getSizeInBytesFor( int numPoints )
{
	const int vectorSize = numPoints*sizeof(hkVector4);
	return vectorSize;
}

void hkFindClosestPositionUtil::setBuffer( void* buffer, int numPoints )
{
	HK_CHECK_ALIGN_REAL(buffer);

	hkVector4* vectorPtr = reinterpret_cast<hkVector4*>(buffer);
	m_positions.setDataUserFree(vectorPtr, 0, numPoints);
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
