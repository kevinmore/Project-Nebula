/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Utils/IndexSet/hkIndexSet.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#ifdef HK_DEBUG
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#endif

hkIndexSet::hkIndexSet()
{
    m_inUpdate = false;
}

hkIndexSet::hkIndexSet(const hkIndexSet& rhs)
{
    HK_ASSERT(0x34243f2, rhs.m_inUpdate == false);
    m_inUpdate = false;
    m_indices = rhs.m_indices;
}

hkBool hkIndexSet::operator==(const hkIndexSet& rhs) const
{
    HK_ASSERT(0x34234, m_inUpdate == false && rhs.m_inUpdate == false);

	const int numIndices = m_indices.getSize();
	if (numIndices != rhs.m_indices.getSize())
	{
		return false;
	}

	return hkString::memCmp(m_indices.begin(), rhs.m_indices.begin(), numIndices * hkSizeOf(int)) == 0;
}


void hkIndexSet::operator=(const hkIndexSet& rhs)
{
    HK_ASSERT(0x34234, m_inUpdate == false && rhs.m_inUpdate == false);
    m_indices = rhs.m_indices;
}

int hkIndexSet::calculateNumIntersecting(const hkIndexSet& rhs) const
{
    int numUnion, numIntersecting;
    calculateNumIntersectAndUnion(*this, rhs, numIntersecting, numUnion);
    return numIntersecting;
}

int hkIndexSet::calculateNumUnion(const hkIndexSet& rhs) const
{
    int numUnion, numIntersecting;
    calculateNumIntersectAndUnion(*this, rhs, numIntersecting, numUnion);
    return numUnion;
}

int hkIndexSet::findIndex(int index) const
{
    int start = 0;
    int end = m_indices.getSize();

	// Use a binary chop to find it - can do as indices are sorted
    while (end - start > 0)
    {
        int middle = (start + end) >> 1;
        int value = m_indices[middle];
        // Chop
        if (index < value)
        {
            end = middle;
        }
        else
        {
			// Found it
			if (value == index)
			{
				return middle;
			}

			// I know its not the middle one, as I just tested that
            start = middle + 1;
        }
    }
    return -1;
}

hkArray<int>& hkIndexSet::startUpdate()
{
    HK_ASSERT(0x3243243, m_inUpdate == false);

    m_inUpdate = true;
    return m_indices;
}

void hkIndexSet::endUpdate()
{
    HK_ASSERT(0xd8279a32, m_inUpdate);
    // Do the sort
    hkSort(m_indices.begin(), m_indices.getSize());

    if (m_indices.getSize() > 0)
    {
        int value = m_indices[0] - 1;

        int* cur = m_indices.begin();
        int* next = cur;
        int* end = m_indices.end();

        for (; next != end; next++)
        {
            if (*next != value)
            {
                value = *next;
                *cur++ = value;
            }
        }

        m_indices.setSizeUnchecked(int(cur - m_indices.begin()));
    }

    m_inUpdate = false;
}

void hkIndexSet::removeIndex(int index)
{
    HK_ASSERT(0x2423432, m_inUpdate);
    // Could be in there multiple times

    const int* cur = m_indices.begin();
    const int* end = m_indices.end();
    int* back = m_indices.begin();

    for (; cur != end; cur++)
    {
        if (*cur != index)
        {
            *back++ = *cur;
        }
    }

    m_indices.setSizeUnchecked(int(back - m_indices.begin()));
}

void hkIndexSet::addIndex(int index)
{
    HK_ASSERT(0x2423432, m_inUpdate);
    m_indices.pushBack(index);
}

/* static */void hkIndexSet::calculateNumIntersectAndUnion(const hkIndexSet& setA, const hkIndexSet& setB, int& numIntersectOut, int& numUnionOut)
{
    HK_ASSERT(0x24324f23, setA.m_inUpdate == false && setB.m_inUpdate == false);

    const int* curA = setA.m_indices.begin();
    const int* endA = setA.m_indices.end();

    const int* curB = setB.m_indices.begin();
    const int* endB = setB.m_indices.end();

    int numIntersect = 0;
    int numNonIntersect = 0;
    while (curA != endA && curB != endB)
    {
        const int a = *curA;
        const int b = *curB;
        if (a == b)
        {
            numIntersect++;
            curA++;
            curB++;
        }
        else
        {
            numNonIntersect++;
            // Stop forward
            if (a < b)
            {
                curA++;
            }
            else
            {
                curB++;
            }
        }
    }

    // Add the remaining
    numNonIntersect += int((endA - curA) + (endB - curB));

    // Output
    numIntersectOut = numIntersect;
    numUnionOut = numIntersect + numNonIntersect;
}

 void hkIndexSet::setIntersection(const hkIndexSet& setA, const hkIndexSet& setB)
{
    HK_ASSERT(0x2c432423, m_inUpdate == false && setA.m_inUpdate == false && setB.m_inUpdate == false);
    HK_ASSERT(0x03423432, this != &setA && this != &setB);

    m_indices.clear();

    const int* curA = setA.m_indices.begin();
    const int* endA = setA.m_indices.end();

    const int* curB = setB.m_indices.begin();
    const int* endB = setB.m_indices.end();

    while (curA != endA && curB != endB)
    {
        const int a = *curA;
        const int b = *curB;
        if (a == b)
        {
            curA++;
            curB++;
            m_indices.pushBack(a);
        }
        else
        {
            if (a < b)
            {
                curA++;
            }
            else
            {
                curB++;
            }
        }
    }
}

void hkIndexSet::setDifference(const hkIndexSet& setA, const hkIndexSet& setB)
{
    HK_ASSERT(0x193be909, m_inUpdate == false && setA.m_inUpdate == false && setB.m_inUpdate == false);
    HK_ASSERT(0x193be908, this != &setA && this != &setB);

    m_indices.clear();
    const int* curA = setA.m_indices.begin();
    const int* endA = setA.m_indices.end();

    const int* curB = setB.m_indices.begin();
    const int* endB = setB.m_indices.end();

    while (curA != endA && curB != endB)
    {
        const int a = *curA;
        const int b = *curB;
        if (a == b)
        {
            curA++;
            curB++;
        }
        else
        {
            if (a < b)
            {
                m_indices.pushBack(a);
                curA++;
            }
            else
            {
                curB++;
            }
        }
    }
	// Add any remaining
	if (curA != endA)
	{
		const int numRemaining = int(endA - curA);
		hkString::memCpy(m_indices.expandBy(numRemaining), curA, numRemaining * sizeof(int));
	}
}

void hkIndexSet::setUnion(const hkIndexSet& setA, const hkIndexSet& setB)
{
    HK_ASSERT(0x2432a423, m_inUpdate == false && setA.m_inUpdate == false && setB.m_inUpdate == false);
    HK_ASSERT(0x342b3432, this != &setA && this != &setB);

    m_indices.clear();
    const int* curA = setA.m_indices.begin();
    const int* endA = setA.m_indices.end();

    const int* curB = setB.m_indices.begin();
    const int* endB = setB.m_indices.end();

    while (curA != endA && curB != endB)
    {
        const int a = *curA;
        const int b = *curB;
        if (a == b)
        {
            curA++;
            curB++;
            m_indices.pushBack(a);
        }
        else
        {
            if (a < b)
            {
                m_indices.pushBack(a);
                curA++;
            }
            else
            {
                m_indices.pushBack(b);
                curB++;
            }
        }
    }

	// Add any remaining
	if (curA != endA)
	{
		const int numRemaining = int(endA - curA);
		hkString::memCpy(m_indices.expandBy(numRemaining), curA, numRemaining * sizeof(int));
	}
	if (curB != endB)
	{
		const int numRemaining = int(endB - curB);
		hkString::memCpy(m_indices.expandBy(numRemaining), curB, numRemaining * sizeof(int));
	}
}

void hkIndexSet::optimizeAllocation()
{
    m_indices.optimizeCapacity( 0, true);
}

#ifdef HK_DEBUG

/* static */void hkIndexSet::selfTest()
{
    hkPseudoRandomGenerator rand(1000);
    hkIndexSet a;

    // Some simple tests
    HK_ASSERT(0x3424234, a == a);
    HK_ASSERT(0x3424234, a.calculateNumIntersecting(a) == 0);
    HK_ASSERT(0x3424234, a.calculateNumUnion(a) == 0);


    {
        hkArray<int>& indices = a.startUpdate();
        for (int i = 0; i < 1000; i++)
        {
            indices.pushBack(rand.getRand32());
        }
        a.endUpdate();
    }

    hkIndexSet b;
    {
        hkArray<int>& indices = b.startUpdate();
        for (int i = 0; i < 1001; i++)
        {
            indices.pushBack(rand.getRand32());
        }
        b.endUpdate();
    }

    {
        // Force the sets to be different
        b.startUpdate();
        b.removeIndex(a.getIndices()[0]);
        b.endUpdate();
    }

    HK_ASSERT(0x4324324, a != b);

    ///
    HK_ASSERT(0x4324324, a == a && b == b);

    // Union
    hkIndexSet aUnionB;
    aUnionB.setUnion(a, b);
    hkIndexSet bUnionA;
    bUnionA.setUnion(b, a);

    // Commutative
    HK_ASSERT(0x34234534, aUnionB == bUnionA);

    hkIndexSet aIntersectB;
    aIntersectB.setIntersection(a, b);
    hkIndexSet bIntersectA;
    bIntersectA.setIntersection(b, a);

    HK_ASSERT(0x34234534, aIntersectB == bIntersectA);

    HK_ASSERT(0x277abd34, aIntersectB.getSize() == a.calculateNumIntersecting(b));
    HK_ASSERT(0x277abd33, aUnionB.getSize() == a.calculateNumUnion(b));

    // Do some checks
    HK_ASSERT(0x2432432, aUnionB.calculateNumIntersecting(aIntersectB) == aIntersectB.getSize());

    hkIndexSet aMinusB;
    aMinusB.setDifference(a, b);
    hkIndexSet bMinusA;
    bMinusA.setDifference(b, a);

    HK_ASSERT(0x2432433, aMinusB.calculateNumIntersecting(a) == aMinusB.getSize());
    HK_ASSERT(0x2432434, aMinusB.calculateNumIntersecting(b) == 0);

    HK_ASSERT(0x2432435, bMinusA.calculateNumIntersecting(b) == bMinusA.getSize());
    HK_ASSERT(0x2432436, bMinusA.calculateNumIntersecting(a) == 0);

    // I should be able to manufacture some

    {
        hkIndexSet test;
        test.setUnion(aMinusB, aIntersectB);

        HK_ASSERT(0xbd838ddb, test == a);
    }

    {
        hkIndexSet test;
        test.setUnion(bMinusA, aIntersectB);
        HK_ASSERT(0xd8279a31, test == b);
    }

    {
        hkIndexSet test;
        test.setUnion(aMinusB, aIntersectB);
        hkIndexSet test2;
        test2.setUnion(test, bMinusA);

        HK_ASSERT(0xd8279a30, test2 == aUnionB);
    }
}

#endif

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
