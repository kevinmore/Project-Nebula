/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Matrix/hkMatrixdNm.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/Math/Vector/hkVectorNd.h>

/* static */const hkVector4ComparisonMask::Mask hkMatrixdNm::s_zeroMask[4] = 
{
	hkVector4ComparisonMask::MASK_XYZW,
	hkVector4ComparisonMask::MASK_X,
	hkVector4ComparisonMask::MASK_XY,
	hkVector4ComparisonMask::MASK_XYZ,
};

hkMatrixdNm::hkMatrixdNm(const hkMatrixdNm& rhs)
{
	m_numCols = rhs.m_numCols;
    m_numRows = rhs.m_numRows;

	m_elements = rhs.m_elements;
}

void hkMatrixdNm::setAdd(const hkMatrixdNm& a, const hkMatrixdNm& b) 
{ 
	HK_ASSERT(0x242342aa, this != &b);
	if (this != &a)
	{
		*this = a; 
	}
	add(b); 
}


void hkMatrixdNm::setSub(const hkMatrixdNm& a, const hkMatrixdNm& b) 
{ 
	HK_ASSERT(0x242342aa, this != &b);
	if (this != &a)
	{
		*this = a; 
	}
	sub(b); 
}

void hkMatrixdNm::setSize(int numRows, int numCols)
{
    if (numCols != m_numCols || numRows != m_numRows)
    {
        _setSize(numRows, numCols);
    }
}

hkMatrixdNm::hkMatrixdNm(int numRows, int numCols, const hkDouble64* values)
{
	m_numCols = numCols;
    m_numRows = numRows;

    if (numRows > 0 && numCols > 0)
    {
        const int colLen = (numRows + 3) >> 2;
        m_elements.setSize(colLen * numCols);

        _setValues(numRows, numCols, values, m_elements.begin());
    }
}

/* static */void hkMatrixdNm::_setValues( int numRows, int numCols, const hkDouble64* v, hkVector4d* dst)
{
    const int colLen = (numRows + 3) >> 2;

    if ((numRows & 3) == 0)
    {
		hkString::memCpy(dst, v, colLen * numCols * sizeof(hkDouble64) * 4);
    }
    else
    {
        // I've got to take into account the stride
		hkVector4dComparison mask; mask.set(s_zeroMask[numRows & 3]);
		hkVector4d zero; zero.setZero();

        for (int i = 0; i < numCols; i++)
        {
            hkVector4d* end = dst + colLen;

            if ((hk_size_t(v) & 0xf) == 0)
            {
                const hkVector4d* src = (const hkVector4d*)v;
                while (dst != end) *dst++ = *src++;
            }
            else
            {
                const hkDouble64* src = v;
                for (; dst != end; dst++, src += 4) dst->load<4,HK_IO_NATIVE_ALIGNED>(src);
            }

            v += numRows;
            hkVector4d& last = dst[-1];

			last.zeroIfFalse(mask);
        }
    }
}

void hkMatrixdNm::setOuterProduct(const hkVectorNd& a, const hkVectorNd& b)
{
	setSize(a.m_size, b.m_size);

	const int numColVecs = (a.m_size + 3) >> 2;

	// lets work out the outer product
	hkVector4d* dst = m_elements.begin();
	for (int i = 0; i < b.m_size; i++)
	{
		hkVector4d v; v.setBroadcast(i & 3, b.m_elements[i >> 2]);
		const hkVector4d* avecs = a.m_elements;

		hkVector4d* end = dst + numColVecs;
		for (; dst < end; dst++, avecs++)
		{
			dst->setMul(v, *avecs);
		}
	}
}

void hkMatrixdNm::set( int numRows, int numCols, const hkDouble64* v)
{
    if (numCols != m_numCols || numRows != m_numRows)
    {
        const int colLen = (numRows + 3) >> 2;
        m_elements.setSize(colLen * numCols);

        m_numRows = numRows;
        m_numCols = numCols;
    }

    _setValues(numRows, numCols, v, m_elements.begin());
}

void hkMatrixdNm::_setSize(int numRows, int numCols)
{
    const int colLen = (numRows + 3) >> 2;
    const int numVecs = colLen * numCols;

    if (numVecs > 0)
    {
        m_elements.setSize(numVecs);

        if (numCols & 3)
        {
            // Clear the final rows
            hkVector4d* const end = m_elements.end();
            hkVector4d* cur = m_elements.begin() + colLen - 1;
            for (; cur < end; cur += colLen)
            {
                cur->setZero();
            }
        }
    }

    m_numCols = numCols;
    m_numRows = numRows;
}

void hkMatrixdNm::setIdentity()
{
    HK_ASSERT(0x23432432, isSquare());

	const int size = m_numCols;
	const int numColVecs = (m_numRows + 3) >> 2;

	hkVector4d ones; ones.setAll(1.0f);

	hkVector4d* dst = m_elements.begin();
	for (int i = 0; i < size; i++)
	{
		const int colVec = i >> 2;
		for (int j = 0; j < numColVecs; j++, dst++)
		{
			if (j == colVec)
			{
				// There is a 1 in this section
				hkVector4d t = ones;
				t.zeroComponent(i & 3);
				dst->setSub(ones, t);
			}
			else
			{
				dst->setZero();
			}
		}
	}
}

hkBool hkMatrixdNm::isIdentity(hkDouble64 epsilon) const
{
    HK_ASSERT(0x2432a423, isSquare());
    if (!isSquare())
    {
        return false;
    }

	const int size = m_numCols;
	const int numColVecs = (m_numRows + 3) >> 2;

	hkVector4d ones; ones.setAll(1.0f);
	hkVector4d zeros; zeros.setZero();
	hkSimdDouble64 sEps; sEps.setFromFloat(epsilon);

	const hkVector4d* src = m_elements.begin();
	for (int i = 0; i < size; i++)
	{
		const int colVec = i >> 2;
		for (int j = 0; j < numColVecs; j++, src++)
		{
			if (j == colVec)
			{
				// There is a 1 in this section
				hkVector4d t = ones;
				t.zeroComponent(i & 3);
				hkVector4d v; v.setSub(ones, t);

				if (!v.allEqual<4>(*src, sEps))
				{
					return false;
				}
			}
			else
			{
				if (!zeros.allEqual<4>(*src, sEps))
				{
					return false;
				}
			}
		}
	}

    return true;
}


void hkMatrixdNm::setTranspose(const hkMatrixdNm& rhs)
{
    HK_ASSERT(0x42342423, this != &rhs);
    setSize(rhs.m_numCols, rhs.m_numRows);

    // Work out the stride
    int srcStride = ((rhs.m_numRows + 3) >> 2) * (sizeof(hkVector4d) / sizeof(hkDouble64));
    int dstStride = ((m_numRows + 3) >> 2) * (sizeof(hkVector4d) / sizeof(hkDouble64));

    hkDouble64* dst = (hkDouble64*)m_elements.begin();

    for (int i = 0; i < m_numCols; i++)
    {
        const hkDouble64* src = ((const hkDouble64*)rhs.m_elements.begin()) + i;

        for (int j = 0; j < m_numRows; j++)
        {
            dst[j] = *src;
            src += srcStride;
        }

        // Down a row
        dst += dstStride;
    }
}

void hkMatrixdNm::add(const hkMatrixdNm& rhs)
{
    HK_ASSERT(0x24234a32, m_numRows == rhs.m_numRows && m_numCols == rhs.m_numCols);

    hkVector4d* dst = m_elements.begin();
    hkVector4d* end = m_elements.end();
    const hkVector4d* src = rhs.m_elements.begin();

    for (; dst != end; dst++, src++)
    {
        dst->add(*src);
    }
}

void hkMatrixdNm::sub(const hkMatrixdNm& rhs)
{
    HK_ASSERT(0x24234a32, m_numRows == rhs.m_numRows && m_numCols == rhs.m_numCols);

    hkVector4d* dst = m_elements.begin();
    hkVector4d* end = m_elements.end();
    const hkVector4d* src = rhs.m_elements.begin();

    for (; dst != end; dst++, src++)
    {
        dst->sub(*src);
    }
}

hkBool hkMatrixdNm::isOk() const
{
    const hkVector4d* const end = m_elements.end();

    {
        const hkVector4d* cur = m_elements.begin();
        for (; cur != end; cur++)
        {
            if (!cur->isOk<4>())
            {
                return false;
            }
        }
    }

    if (m_numRows & 3)
    {
        // Need to checks zeros
        const int colLen = (m_numRows + 3) >> 2;
        const hkVector4d* cur = m_elements.begin() + colLen - 1;

		hkVector4d zero; zero.setZero();
		hkVector4dComparison mask; mask.set(s_zeroMask[m_numRows & 3]);

        for (; cur < end; cur += colLen)
        {
			hkVector4d v;
			v.setSelect( mask, *cur, zero);
			hkSimdDouble64 eps; eps.setFromFloat(1e-3f);
            if (!v.allEqual<4>(*cur,eps))
            {
                return false;
            }
        }
    }
    return true;
}

hkBool hkMatrixdNm::equals(const hkMatrixdNm& rhs, const hkDouble64 threshold) const
{
    if (m_numCols != rhs.m_numCols ||
        m_numRows != rhs.m_numRows)
    {
        return false;
    }

    const hkVector4d* a = m_elements.begin();
    const hkVector4d* end = m_elements.end();
    const hkVector4d* b = rhs.m_elements.begin();

	hkSimdDouble64 st; st.setFromFloat(threshold);
    for (; a != end; a++, b++)
    {
        if (!a->allEqual<4>(*b, st))
        {
            return false;
        }
    }

    return true;
}

void hkMatrixdNm::mul(hkSimdDouble64Parameter v)
{
    hkVector4d* dst = m_elements.begin();
    hkVector4d* end = m_elements.end();

    for (; dst != end; dst++)
    {
        dst->mul(v);
    }
}

void hkMatrixdNm::getRow(int row, hkVectorNd& rowOut) const
{
    HK_ASSERT(0x3434a234, row >= 0 && row < m_numRows);

    rowOut.setSize(m_numCols);
    const hkDouble64* src = ((const hkDouble64*)m_elements.begin()) + row;
    int stride = getRealStride();
    hkDouble64* dst = rowOut.getElements();

    for (int i = 0; i < m_numCols; i++)
    {
        dst[i] = *src;
        src += stride;
    }
}


void hkMatrixdNm::getColumn(int col, hkVectorNd& colOut) const
{
    HK_ASSERT(0x32423432, col >= 0 && col < m_numCols);

    if (colOut.m_size != m_numRows)
    {
        colOut._setSize(m_numRows);
    }

    // Copy the row
    const int colLen = (m_numRows + 3) >> 2;

    const hkVector4d* src = m_elements.begin() + (col * colLen);
    hkVector4d* dst = colOut.m_elements;

    for (int i = 0; i < colLen; i++)
    {
        dst[i] = src[i];
    }
}

void hkMatrixdNm::getColumnAlias(int col, hkVectorNd& colOut)
{
    HK_ASSERT(0x4242a34a, col >= 0 && col < m_numCols);

    const int colLen = (m_numRows + 3) >> 2;
    hkVector4d* src = m_elements.begin() + (col * colLen);
    colOut.alias(src, m_numRows);
}

void hkMatrixdNm::multiply(const hkVectorNd& in, hkVectorNd& out) const
{
    HK_ASSERT(0x41432ab0, in.getSize() == m_numCols);
    out.setSize(m_numRows);

    // Work out the transformed
    const int numVecs = (m_numRows + 3) >> 2;

    const int numComplete = m_numCols >> 2;
    const int numPartial =  m_numCols & 3;

    hkVector4d* dst = out.getVectors();

    for (int i = 0; i < numVecs; i++)
    {
        const hkVector4d* src = in.getVectors();
        const hkVector4d* m = m_elements.begin() + i;

        hkVector4d t; // temp
        hkVector4d total; total.setZero();

        for (int j = 0; j < numComplete; j++)
        {
            t.setBroadcast<0>(*src);
            total.addMul(t, *m);
            m += numVecs;

            t.setBroadcast<1>(*src);
            total.addMul(t, *m);
            m += numVecs;

            t.setBroadcast<2>(*src);
            total.addMul(t, *m);
            m += numVecs;

            t.setBroadcast<3>(*src);
            total.addMul(t, *m);
            m += numVecs;

            src++;
        }

        // Need to do the partial ones
        switch (numPartial)
        {
            case 3:
                t.setBroadcast<2>(*src);
                total.addMul(t, m[numVecs * 2]);
            case 2:
                t.setBroadcast<1>(*src);
                total.addMul(t, m[numVecs * 1]);
            case 1:
                t.setBroadcast<0>(*src);
                total.addMul(t, m[numVecs * 0]);
            default: break;
        }

        // Save the total
        *dst++ = total;
    }
}

void hkMatrixdNm::setFpuMul(const hkMatrixdNm& a, const hkMatrixdNm& b)
{
    HK_ASSERT(0x3243a242, this != &a && this != &b);
    HK_ASSERT(0x5345435a, a.m_numCols == b.m_numRows);

    // Make sure the target has enough space
    setSize(a.m_numRows, b.m_numCols);

	for (int i = 0; i < b.m_numCols; i++)
	{
		for (int j = 0; j < a.m_numRows; j++)
		{
			hkDouble64 total = 0.0f;
			for (int k = 0; k < b.m_numRows; k++)
			{
				total += a(j, k) * b(k, i);
			}

			(*this)(j, i) = total;
		}
	}
}


void hkMatrixdNm::setMul(const hkMatrixdNm& a, const hkMatrixdNm& b)
{
    HK_ASSERT(0x3243a242, this != &a && this != &b);
    HK_ASSERT(0x5345435a, a.m_numCols == b.m_numRows);

    // Make sure the target has enough space
	setSize(a.m_numRows, b.m_numCols);

    const int numComplete = b.m_numRows >> 2;
    const int numPartial =  b.m_numRows & 3;

    const hkVector4d* colB = b.m_elements.begin();

    // Work out the row strides
    const int strideA = (a.m_numRows + 3) >> 2;
    const int strideB = (b.m_numRows + 3) >> 2;

    hkVector4d* dst = m_elements.begin();
    for (int k = 0; k < b.m_numCols; k++)
    {
        for (int i = 0; i < strideA; i++)
        {
            const hkVector4d* srcA = a.m_elements.begin() + i;
            const hkVector4d* srcB = colB;

            hkVector4d t; // temp
            hkVector4d total; total.setZero();

            for (int j = 0; j < numComplete; j++)
            {
                hkVector4d vb = *srcB;

                t.setBroadcast<0>(vb);
                total.addMul(t, *srcA);
                srcA += strideA;

                t.setBroadcast<1>(vb);
                total.addMul(t, *srcA);
                srcA += strideA;

                t.setBroadcast<2>(vb);
                total.addMul(t, *srcA);
                srcA += strideA;

                t.setBroadcast<3>(vb);
                total.addMul(t, *srcA);
                srcA += strideA;

                srcB++;
            }

            if (numPartial)
            {
                hkVector4d vb = *srcB;
                // Need to do the partial ones
                switch (numPartial)
                {
                    case 3:
                        t.setBroadcast<2>(vb);
                        total.addMul(t, srcA[strideA * 2]);
                    case 2:
                        t.setBroadcast<1>(vb);
                        total.addMul(t, srcA[strideA * 1]);
                    case 1:
                        t.setBroadcast<0>(vb);
                        total.addMul(t, srcA[strideA * 0]);
                    default: break;
                }
            }

            // Save the total
            *dst++ = total;
        }

        // dst just wraps around to the next line
        colB += strideB;
    }
}

// This invert() function below was written to take advantage of the masks defined in hkVector4dComparison
// for the SSE build.  But the other builds use different masks, so here we map them. See hkSseMathTypes.inl.
static hkUint8 g_pivotMaskMap[16] =
{
	hkVector4ComparisonMask::MASK_NONE,
	hkVector4ComparisonMask::MASK_X,
	hkVector4ComparisonMask::MASK_Y,
	hkVector4ComparisonMask::MASK_XY,
	hkVector4ComparisonMask::MASK_Z,
	hkVector4ComparisonMask::MASK_XZ,
	hkVector4ComparisonMask::MASK_YZ,
	hkVector4ComparisonMask::MASK_XYZ,
	hkVector4ComparisonMask::MASK_W,
	hkVector4ComparisonMask::MASK_XW,
	hkVector4ComparisonMask::MASK_YW,
	hkVector4ComparisonMask::MASK_XYW,
	hkVector4ComparisonMask::MASK_ZW,
	hkVector4ComparisonMask::MASK_XZW,
	hkVector4ComparisonMask::MASK_YZW,
	hkVector4ComparisonMask::MASK_XYZW,
};

hkResult hkMatrixdNm::invert()
{
    HK_ASSERT(0x245a25a4, isSquare());

    const int floatStride = getRealStride();

    const int size = m_numCols;
    const int numVecs = (size + 3) >> 2;

    const int inplaceSize = 32;

    hkInplaceArray<int, inplaceSize> colIndex; colIndex.setSize(size);
    hkInplaceArray<int, inplaceSize> rowIndex; rowIndex.setSize(size);

    const int inplacePivotSize = (inplaceSize + 7) >> 3;
    hkInplaceArray<hkUint8, inplacePivotSize> pivoted; pivoted.setSize(numVecs, 0);
    hkInplaceArray<hkDouble64*, inplaceSize> cols; cols.setSize(size);

    // Set up the rows
    {
        hkDouble64* cur = (hkDouble64*)m_elements.begin();

        for (int i = 0; i < size; i++)
        {
            cols[i] = cur;
            cur += floatStride;
        }

		HK_ASSERT(0x4424234, cur == (hkDouble64*)m_elements.end());
    }

    // Work out the shift
    
    // We want to add 1 if its not a power of 2
    int maxShift = 0;
	while ( numVecs > (1 << maxShift)) maxShift++;

	// Add 2 - as its in floats (numVecs is in vec4s)
	maxShift += 2;

    // The stride between elements from the max calculation - rounded to a power of 2, to make extraction of col/row fast
    const int maxStride = 1 << maxShift;

    // Set up constants
    hkVector4d indicesStep; indicesStep.setAll(4.0f);
    hkVector4d skipColStep; skipColStep.setAll(hkDouble64(maxStride));
    hkVector4d remainColStep; remainColStep.setAll(hkDouble64(maxStride - (numVecs * 4)));
	hkVector4d zero; zero.setZero();

    // elimination by full pivoting
    for (int i = 0; i < size; i++)
    {
        hkVector4d max; max.setZero();
        hkVector4d maxIndices = hkVector4d::getConstant<HK_QUADREAL_MINUS1>();

        hkVector4d indices; indices.set(0.25f, 1.25f, 2.25f, 3.25f);

        for (int j = 0; j < size; j++)
        {
            if ((pivoted[j >> 2] & (1 << (j & 3))) == 0)
            {
                // Work out the max
                const hkVector4d* vecCol = (const hkVector4d*)cols[j];

                for (int k = 0; k < numVecs; k++)
                {
					int pivotMask = pivoted[k];
					if (pivotMask != 0xf)
					{
						hkVector4d t; t.setAbs(vecCol[k]);

						// The is where we use the fact that bit 1 corresponds to MASK_X etc, so we have to apply the pivot mask map.
						hkVector4dComparison mask; mask.set(hkVector4ComparisonMask::Mask(g_pivotMaskMap[pivotMask]));
						t.zeroIfTrue(mask);

	                    hkVector4dComparison cmp = t.greater(max);

	                    max.setSelect(cmp, t, max);
		                maxIndices.setSelect( cmp, indices, maxIndices );
					}
                    
                    indices.add(indicesStep);
                }
                indices.add(remainColStep);
            }
            else
            {
                indices.add(skipColStep);
            }
        }

		// Find the max index 
		int index = 0;
		for (int j = 1; j < 4; j++)
		{
			if (max(j) > index)
			{
				index = j;
			}
		}

        const hkDouble64 rowColIndexReal = maxIndices(index);
        if (rowColIndexReal < 0.0f)
        {
            return HK_FAILURE;
        }

		const int colRowIndex = hkMath::hkToIntFast(rowColIndexReal);

        // I could use a shift
        const int col = colRowIndex >> maxShift;
        const int row = colRowIndex & (maxStride - 1);

        // Mark as pivoted
        pivoted[row >> 2] |= (1 << (row & 3));

        // swap rows so that a[row][row] contains the pivot entry
        if (row != col)
        {
            hkDouble64* swap = cols[row];
            cols[row] = cols[col];
            cols[col] = swap;
        }

        // keep track of the permutations of the rows
        rowIndex[i] = row;
        colIndex[i] = col;

        // scale the row so that the pivot entry is 1
		hkDouble64 inv = 1.0f / cols[row][row];

        cols[row][row] = 1.0f;

        {
            hkVector4d* colVecs = (hkVector4d*)(cols[row]);
            hkVector4d t; t.setAll(inv);
            for (int j = 0; j < numVecs; j++)
            {
                colVecs[j].mul(t);
            }
        }

        // zero out the pivot column locations in the other cols
        for (int j = 0; j < size; j++)
        {
            if (j != row)
            {
                hkDouble64* dstCol = cols[j];

                hkVector4d t; t.setAll(-dstCol[row]);
                dstCol[row] = 0.0f;

                hkVector4d* dstVecCol = (hkVector4d*)dstCol;
                const hkVector4d* srcVecCol = (const hkVector4d*)cols[row];
                for (int k = 0; k < numVecs; k++)
                {
                    dstVecCol[k].addMul(srcVecCol[k], t);
                }
            }
        }
    }

    // reorder cols so that A[][] stores the inverse of the original matrix
    {
        for (int i = size - 1; i >= 0; i--)
        {
            if (rowIndex[i] != colIndex[i])
            {
                // Swap the columns
                hkDouble64* a = ((hkDouble64*)m_elements.begin()) + rowIndex[i];
                hkDouble64* b = ((hkDouble64*)m_elements.begin()) + colIndex[i];
                for (int j = 0; j < size; j++, a += floatStride, b += floatStride)
                {
                    hkDouble64 swap = *a;
                    *a = *b;
                    *b = swap;
                }
            }
        }
    }

    // Need to reorder based on the col pointers
    {
        hkArray<hkVector4d> work;
        work.setSize(m_elements.getSize());

        // Copy the rows over
        {
            hkVector4d* dst = work.begin();

            for (int i = 0; i < size; i++)
            {
                const hkVector4d* src = (hkVector4d*)cols[i];
                // Copy the col
                for (int j = 0; j < numVecs; j++)
                {
                    dst[j] = src[j];
                }
                // Down a row
                dst += numVecs;
            }
        }
        m_elements.swap(work);
    }

    return HK_SUCCESS;
}

hkBool hkMatrixdNm::isSymmetric(hkDouble64 threshold) const
{
	if (m_numCols != m_numRows)
	{
		return false;
	}

	const int size = m_numCols;
	const int realStride = getRealStride();
	const hkDouble64* t1 = ((const hkDouble64*)m_elements.begin()) + 1;
	const hkDouble64* t2 = ((const hkDouble64*)m_elements.begin()) + realStride;

	for (int i = 1; i < size; i++)
	{
		const hkDouble64* c1 = t1;
		const hkDouble64* c2 = t2;
		for (int j = i; j < size; j++)
		{
			hkDouble64 err = hkMath::fabs(*c1 - *c2);
			if (err > threshold)
			{
				return false;
			}

			c1 ++;
			c2 += realStride;
		}

		t1 += realStride + 1;
		t2 += realStride + 1;
	}

	return true;
}



#ifdef HK_DEBUG

/* static */void hkMatrixdNm::selfTest()
{
    {
        hkPseudoRandomGenerator rand(100);

		for (int j = 0; j < 100; j++)
		{
			const int size = (rand.getRand32() % 100) + 2;
			hkMatrixdNm a(size, size);
			for (int i = 0; i < size * size; i++)
			{
				const int v = (rand.getRand32() % 200) - 100;
				a(i / size, i % size) = hkDouble64(v);
			}

			hkMatrixdNm invert(a);
			if (invert.invert() == HK_SUCCESS)
			{
				hkMatrixdNm r;

				r.setMul(a, invert);
				//r.setFpuMul(a, invert);

				HK_ASSERT(0x432a4324, r.isIdentity(1e-2f));
			}
		}
    }

	{
		hkMatrixdNm m;
		m.setSize(5, 5);
		m.setIdentity();

		HK_ASSERT(0x324a3432, m.isSymmetric() );

		m(2, 4) = 3.0f;
		HK_ASSERT(0x324a3432, !m.isSymmetric() );
	}

	{
        hkPseudoRandomGenerator rand(100);

		for (int k = 0; k < 100; k++)
		{
			const int numRowsA = (rand.getRand32() % 100) + 1;
			const int numColsA = (rand.getRand32() % 100) + 1;
			const int numRowsB = numColsA;
			const int numColsB = (rand.getRand32() % 100) + 1;

			hkMatrixdNm a(numRowsA, numColsA);
			hkMatrixdNm b(numRowsB, numColsB);

			for (int i = 0; i < numRowsA; i++)
			{
				for (int j = 0; j < numColsA; j++)
				{
					const int v = (rand.getRand32() % 20) - 10;
					a(i, j) = hkDouble64(v);
				}
			}

			for (int i = 0; i < numRowsB; i++)
			{
				for (int j = 0; j < numColsB; j++)
				{
					const int v = (rand.getRand32() % 20) - 10;
					b(i, j) = hkDouble64(v);
				}
			}

			hkMatrixdNm fpuR;
			hkMatrixdNm r;

			fpuR.setFpuMul(a, b);
			r.setMul(a, b);

			HK_ASSERT(0x3242a423, r.equals(fpuR));
		}
	}

    {
        const hkDouble64 vals[] = { 1, 2, 3, 4, 5, 6 };
        hkMatrixdNm a(2, 3, vals);
        hkMatrixdNm b; b.setTranspose(a);
        hkMatrixdNm c; c.setTranspose(b);

        HK_ASSERT(0x432a4a23, a.equals(c));

        hkVectorNd v;
        a.getRow(0, v);
        HK_ASSERT(0x423432a4, v.equalsStrided(vals + 0, 3, 2));

        a.getRow(1, v);
        HK_ASSERT(0x3242343a, v.equalsStrided(vals + 1, 3, 2));

        a.getColumnAlias(1, v);
        HK_ASSERT(0x43242343, v.equalsStrided(vals + 2, 2, 1));
        v.unalias();

        a.getColumn(0, v);
        HK_ASSERT(0x43242343, v.equalsStrided(vals + 0, 2, 1));

        HK_ASSERT(0x43242343, v.horizontalAdd().getReal() == (1.0f + 2));
    }

	{
		
        const hkDouble64 vals[] = { 1, 2, 3, 4, 5, 6 };
        hkMatrixdNm mat(2, 3, vals);

		const hkDouble64 avals[] = { -1, 2, 3 };
		const hkDouble64 rvals[] = { 20, 24 };
		hkVectorNd a(avals, 3);
		hkVectorNd r;

		mat.multiply(a, r);

		HK_ASSERT(0x244324aa, r.equalsStrided(rvals, 2, 1));
	}

	{
		const hkDouble64 avals[] = 
		{
			1, 2, 3, 4, 5, 6
		};
		const hkDouble64 bvals[] = 
		{
			-1, 3, -1, -2, 3, 1
		};
		const hkDouble64 rvals[] = { 3, 4, 12, 14 };

		hkMatrixdNm a(2, 3, avals);
		hkMatrixdNm b(3, 2, bvals);

		hkMatrixdNm r;
		r.setMul(a, b);

		hkMatrixdNm t(2, 2, rvals);

		HK_ASSERT(0x3242a423, t.equals(r));

	}

	{
        HK_ALIGN_DOUBLE(const hkDouble64 mvals[]) =
        {
            1, 4, -2, 4,
            0, 2, 3, 0,
            -3, 4, 4, 0,
            -1, 2, 3, 5
        };
        HK_ALIGN_DOUBLE(const hkDouble64 vvals[]) =
        {
            1, 4, 0, 2
        };

        hkMatrixdNm m(4, 4, mvals);
        
        hkVectorNd v(vvals, 4);
        hkVectorNd r;

        m.multiply(v, r);

        // See if same result
        hkMatrix4d mat; mat.set4x4ColumnMajor(mvals);
        hkVector4d vec; vec.load<4>(vvals);
        hkVector4d rvec;

		mat.multiplyVector(vec, rvec);

        HK_ASSERT(0x4234a324, !m.isIdentity());

        // See if the same result
		hkSimdDouble64 eps; eps.setFromFloat(1e-3f);
        HK_ASSERT(0x535111a1, rvec.allEqual<4>(*r.getVectors(),eps));

        // For a laugh try inverting
        hkMatrixdNm inverse(m);
        inverse.invert();

        hkMatrixdNm test;
        test.setMul(m, inverse);

        HK_ASSERT(0x42423aa4, test.isIdentity());

        m.setIdentity();
        HK_ASSERT(0x42423a43, m.isIdentity());
    }

	{
		const hkDouble64 avals[] = { 1, 2, 3, 4 };
		const hkDouble64 bvals[] = { 2, -1 };
		const hkDouble64 rvals[] = { 2, 4, 6, 8, -1, -2, -3, -4 };

		hkVectorNd a(avals, HK_COUNT_OF(avals));
		hkVectorNd b(bvals, HK_COUNT_OF(bvals));

		hkMatrixdNm r(4, 2, rvals);
		hkMatrixdNm m; m.setOuterProduct(a, b);

		HK_ASSERT(0x242a4323, r.equals(m));

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
