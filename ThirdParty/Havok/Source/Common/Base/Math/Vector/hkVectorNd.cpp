/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Vector/hkVectorNd.h>

/* static */const hkVector4ComparisonMask::Mask hkVectorNd::s_zeroMask[4] = 
{
	hkVector4ComparisonMask::MASK_XYZW,
	hkVector4ComparisonMask::MASK_X,
	hkVector4ComparisonMask::MASK_XY,
	hkVector4ComparisonMask::MASK_XYZ,
};

hkVectorNd::hkVectorNd(int size):
    m_size(size),
    m_flags(0)
{
    if (size > 0)
    {
        int numEle = (m_size + 3) >> 2;
#if !defined(HK_PLATFORM_PS3_SPU)
        m_elements = hkAlignedAllocate<hkVector4d>(HK_DOUBLE_ALIGNMENT, numEle, HK_MEMORY_CLASS_MATH);
#else
		m_elements = hkAllocateChunk<hkVector4d>( numEle, HK_MEMORY_CLASS_MATH);
#endif
		// Zero out last members
        m_elements[numEle - 1].setZero();
        m_flags = IS_ALLOCATED;
    }
}

hkVectorNd::hkVectorNd(const hkDouble64* v, int size):
    m_size(size),
    m_flags(0)
{
    if (size > 0)
    {
        const int numEle = (m_size + 3) >> 2;
#if !defined(HK_PLATFORM_PS3_SPU)
		m_elements = hkAlignedAllocate<hkVector4d>(HK_DOUBLE_ALIGNMENT, numEle, HK_MEMORY_CLASS_MATH);
#else
		m_elements = hkAllocateChunk<hkVector4d>( numEle, HK_MEMORY_CLASS_MATH);
#endif
        _setValues(m_elements, v, size);
        m_flags = IS_ALLOCATED;
    }
}

void hkVectorNd::set(const hkDouble64* v, int size)
{
    _setSize(size);
    if (size > 0)
    {
        _setValues(m_elements, v, size);
    }
}

/* static */void hkVectorNd::_setValues(hkVector4d* dst, const hkDouble64* v, int size)
{
    const int numEle = (size + 3) >> 2;

    hkVector4d* end = dst + numEle;

    if ((hk_size_t(v) & (HK_DOUBLE_ALIGNMENT - 1)) == 0)
    {
        // Aligned
		for (; dst != end; dst++, v += 4) 
		{
			dst->load<4,HK_IO_SIMD_ALIGNED>(v);
		}
    }
    else
    {
        // Unaligned
        for (; dst != end; dst++, v += 4) 
		{
			dst->load<4,HK_IO_NATIVE_ALIGNED>(v);
		}
    }
    // Zero the end
    hkVector4d& last = end[-1];

	hkVector4dComparison mask; mask.set(s_zeroMask[size & 3]);
	last.zeroIfFalse(mask);
}

void hkVectorNd::_setSizeAndZeroLast(int size)
{
    _setSize(size);
    if (size & 3)
    {
        const int numEle = (m_size + 3) >> 2;
        hkVector4d& last = m_elements[numEle - 1];

		hkVector4dComparison mask; mask.set(s_zeroMask[size & 3]);
		last.zeroIfFalse(mask);
    }
}

void hkVectorNd::setSizeAndZero(int size)
{
    _setSize(size);
    setZero();
}

void hkVectorNd::setZero()
{
    const int numEle = (m_size + 3) >> 2;

    hkVector4d* dst = m_elements;
    hkVector4d* end = dst + numEle;

    // Zero it
    for (; dst != end; dst++) 
	{
		dst->setZero();
	}
}

void hkVectorNd::negate()
{
	const int numEle = (m_size + 3) >> 2;

	hkVector4d* dst = m_elements;
	hkVector4d* end = dst + numEle;

	// Neg it
	for (; dst != end; dst++) 
	{
		dst->setNeg<4>(*dst);
	}
}

void hkVectorNd::_setSize(int size)
{
    int numEle = (m_size + 3) >> 2;
    int numNewEle = (size + 3) >> 2;

    if (numEle != numNewEle)
    {
		if (!isAllocated())
		{
			if (numNewEle > numEle)
			{
				// If its bigger, then I need to expand the allocation
#if !defined(HK_PLATFORM_PS3_SPU)
				m_elements = hkAlignedAllocate<hkVector4d>(HK_DOUBLE_ALIGNMENT, numNewEle, HK_MEMORY_CLASS_MATH);
#else
				m_elements = hkAllocateChunk<hkVector4d>( numNewEle, HK_MEMORY_CLASS_MATH);
#endif
				m_flags = IS_ALLOCATED;
			}
		}
		else
		{
			// Free old if any are set
			if (numEle)
			{
#if !defined(HK_PLATFORM_PS3_SPU)
				hkAlignedDeallocate<hkVector4d>( m_elements );
#else
				hkDeallocateChunk<hkVector4d>( m_elements, numEle, HK_MEMORY_CLASS_MATH);
#endif
				m_flags = 0;
			}
			// Allocate new if need space
			if (numNewEle)
			{
#if !defined(HK_PLATFORM_PS3_SPU)
				m_elements = hkAlignedAllocate<hkVector4d>(HK_DOUBLE_ALIGNMENT, numNewEle, HK_MEMORY_CLASS_MATH);
#else
				m_elements = hkAllocateChunk<hkVector4d>( numNewEle, HK_MEMORY_CLASS_MATH);
#endif
				m_flags = IS_ALLOCATED;
			}
		}
    }
    // Set the size
    m_size = size;
}

hkVectorNd::hkVectorNd(const hkVectorNd& rhs):
    m_size(rhs.m_size),
    m_flags(0)
{
    if (rhs.m_size)
    {
        const int numEle = (rhs.m_size + 3) >> 2;

#if !defined(HK_PLATFORM_PS3_SPU)
		m_elements = hkAlignedAllocate<hkVector4d>(HK_DOUBLE_ALIGNMENT, numEle, HK_MEMORY_CLASS_MATH);
#else
		m_elements = hkAllocateChunk<hkVector4d>( numEle, HK_MEMORY_CLASS_MATH);
#endif

        hkVector4d* dst = m_elements;
        const hkVector4d* src = rhs.m_elements;
        const hkVector4d* end = dst + numEle;
        // Copy
        while (dst != end) *dst++ = *src++;

        m_flags = IS_ALLOCATED;
    }
}

void hkVectorNd::operator=(const hkVectorNd& rhs)
{
    _setSize(rhs.m_size);

    // Copy
    {
        const int numEle = (rhs.m_size + 3) >> 2;
        hkVector4d* dst = m_elements;
        const hkVector4d* src = rhs.m_elements;
        const hkVector4d* end = dst + numEle;

        while (dst != end) *dst++ = *src++;
    }
}

hkSimdDouble64 hkVectorNd::lengthSquared() const
{
    HK_ASSERT(0x23432432, m_size > 0);

    int numEle = (m_size + 3) >> 2;

    const hkVector4d* src = m_elements;
    const hkVector4d* end = src + numEle;

    hkVector4d total;
    total.setMul(*src, *src);
    src++;

    for (; src != end; src++)
    {
        total.addMul(*src, *src);
    }
    return total.horizontalAdd<4>();
}

hkSimdDouble64 hkVectorNd::length() const
{
	return lengthSquared().sqrt();
}


hkSimdDouble64 hkVectorNd::dot(const hkVectorNd& rhs) const
{
    HK_ASSERT(0x4234a324, m_size > 0 && m_size == rhs.m_size);

    int numEle = (m_size + 3) >> 2;

    const hkVector4d* a = m_elements;
    const hkVector4d* b = rhs.m_elements;

    hkVector4d total;

    total.setMul(a[0], b[0]);
    for (int i = 1; i < numEle; i++)
    {
        total.addMul(a[i], b[i]);
    }

    return total.horizontalAdd<4>();
}

hkSimdDouble64 hkVectorNd::horizontalAdd() const
{
    hkVector4d total; total.setZero();

    int numEle = (m_size + 3) >> 2;
    const hkVector4d* a = m_elements;
    for (int i = 0; i < numEle; i++)
    {
        total.add(a[i]);
    }
    return total.horizontalAdd<4>();
}

hkBool hkVectorNd::equals(const hkVectorNd& rhs, hkDouble64 epsilon) const
{
    HK_ASSERT(0x2432a432, m_size > 0 && m_size == rhs.m_size);
    if (m_size <= 0 || m_size != rhs.m_size)
    {
        return false;
    }

    int numEle = (m_size + 3) >> 2;

    const hkVector4d* a = m_elements;
    const hkVector4d* b = rhs.m_elements;

	hkSimdDouble64 sEps; sEps.setFromFloat(epsilon);
    for (int i = 0; i < numEle; i++)
    {
        if (!a[i].allEqual<4>(b[i], sEps))
        {
            return false;
        }
    }
    return true;
}

hkBool hkVectorNd::equalsStrided(const hkDouble64* ele, int size, int stride, hkDouble64 epsilon) const
{
    if (size != m_size)
    {
        return false;
    }

    const hkDouble64* a = (const hkDouble64*)m_elements;
    for (int i = 0; i < size; i++)
    {
		if (hkMath::fabs(a[i] - *ele) > epsilon)
        {
            return false;
        }
        ele += stride;
    }

    return true;
}

hkBool hkVectorNd::isOk() const
{
    int numEle = (m_size + 3) >> 2;

    for (int i = 0; i < numEle; i++)
    {
        if (!m_elements[i].isOk<4>())
        {
            return false;
        }
    }

    if (m_size & 3)
    {
        // Make sure the last members are all zeros
        hkVector4d a;
        const hkVector4d& last = m_elements[numEle - 1];
		
		hkVector4dComparison mask; mask.set(s_zeroMask[m_size & 3]);

		a.setSelect(mask, last, hkVector4d::getZero());

        // All should be exactly equal
		hkSimdDouble64 eps; eps.setFromFloat(1e-3f);
        return a.allEqual<4>(last,eps) != hkFalse32;
    }

    return true;
}

void hkVectorNd::alias(hkVector4d* vecs, int size)
{
    // If it has allocated memory, then free that memory
    if (m_flags & IS_ALLOCATED)
    {
#if !defined(HK_PLATFORM_PS3_SPU)
		hkAlignedDeallocate<hkVector4d>( m_elements );
#else
		hkDeallocateChunk<hkVector4d>( m_elements, getNumVectors(), HK_MEMORY_CLASS_MATH);
#endif
    }

    m_elements = vecs;
    m_size = size;
    m_flags = 0;

    if(!isOk())
	{
		HK_WARN(0x92fba014, "Illegal alias operation");
	}
}

void hkVectorNd::unalias()
{
    if ((m_flags & IS_ALLOCATED) == 0 && m_size > 0)
    {
        const int numEle = (m_size + 3) >> 2;

        const hkVector4d* src = m_elements;
#if !defined(HK_PLATFORM_PS3_SPU)
        hkVector4d* dst = hkAlignedAllocate<hkVector4d>(HK_DOUBLE_ALIGNMENT, numEle, HK_MEMORY_CLASS_MATH);
#else
		hkVector4d* dst = hkAllocateChunk<hkVector4d>( numEle, HK_MEMORY_CLASS_MATH);
#endif

        // Copy
        for (int i = 0; i < numEle; i++)
        {
            dst[i] = src[i];
        }
        m_elements = dst;
        // Mark as allocated
        m_flags |= IS_ALLOCATED;
    }
}

void hkVectorNd::add(const hkVectorNd& a)
{
	HK_ASSERT(0x5435a435, getSize() > 0 && a.getSize() == getSize());

	const hkVector4d* avec = a.m_elements;
	hkVector4d* dst = m_elements;
	hkVector4d* end = m_elements + ((m_size + 3) >> 2);

	for (; dst < end; dst++, avec++)
	{
		dst->add(*avec);
	}
}

void hkVectorNd::sub(const hkVectorNd& a)
{
	HK_ASSERT(0x5435a435, getSize() > 0 && a.getSize() == getSize());

	const hkVector4d* avec = a.m_elements;
	hkVector4d* dst = m_elements;
	hkVector4d* end = m_elements + ((m_size + 3) >> 2);

	for (; dst < end; dst++, avec++)
	{
		dst->sub(*avec);
	}
}

void hkVectorNd::setSub(const hkVectorNd& a, const hkVectorNd& b)
{
	const int size = a.getSize();
	HK_ASSERT(0x5435a435, size > 0 && b.getSize() == size);
	setSize(size);

	const hkVector4d* avec = a.m_elements;
	const hkVector4d* bvec = b.m_elements;
	hkVector4d* dst = m_elements;
	hkVector4d* end = m_elements + ((m_size + 3) >> 2);

	for (; dst < end; dst++, avec++, bvec++)
	{
		dst->setSub(*avec, *bvec);
	}
}

void hkVectorNd::setAdd(const hkVectorNd& a, const hkVectorNd& b)
{
	const int size = a.getSize();
	HK_ASSERT(0x5435a435, size > 0 && b.getSize() == size);
	setSize(size);

	const hkVector4d* avec = a.m_elements;
	const hkVector4d* bvec = b.m_elements;
	hkVector4d* dst = m_elements;
	hkVector4d* end = m_elements + ((m_size + 3) >> 2);

	for (; dst < end; dst++, avec++, bvec++)
	{
		dst->setAdd(*avec, *bvec);
	}
}

void hkVectorNd::mul(hkSimdDouble64Parameter vIn)
{
	hkVector4d* dst = m_elements;
	hkVector4d* end = m_elements + ((m_size + 3) >> 2);

	for (; dst < end; dst++)
	{
		dst->mul(vIn);
	}
}

void hkVectorNd::setMul(hkSimdDouble64Parameter s, const hkVectorNd& v)
{
	setSize(v.m_size);

	hkVector4d* dst = m_elements;
	hkVector4d* end = m_elements + ((m_size + 3) >> 2);
	hkVector4d* a = v.m_elements;

	for (; dst < end; dst++, a++)
	{
		dst->setMul(*a, s);
	}
}

hkSimdDouble64 hkVectorNd::normalize()
{
	HK_ASSERT(0x32432432, m_size > 0);
	const hkSimdDouble64 len = length();
	if (len.getReal() > 1e-10f)
	{
		hkSimdDouble64 invLen; invLen.setReciprocal(len);
		mul(invLen);
	}
	else
	{
		const int numVecs = (m_size + 3) >> 2;
		m_elements->set(1, 0, 0, 0);

		hkVector4d* dst = m_elements + 1;
		hkVector4d* end = m_elements + numVecs;
		for (; dst < end; dst++)
		{
			dst->setZero();
		}
	}
	return len;
}

/* static */void hkVectorNd::calcUnitAxes(const hkVectorNd& a, const hkVectorNd& b, const hkVectorNd& c, hkVectorNd& e1, hkVectorNd& e2)
{
	HK_ON_DEBUG( const int size = a.getSize(); )
	HK_ASSERT(0x452453a4, size > 0 && size == b.getSize() && size == c.getSize());

	e1.setSub(b, a);
	e1.normalize();

	e2.setSub(c, a);
	
	hkVectorNd t;
	t.setMul(e1.dot(e2), e1);

	e2.sub(t);
	e2.normalize();
}

/* static */hkSimdDouble64 hkVectorNd::calcPlaneDistanceSquared(const hkVectorNd& p, hkVectorNd& e1, hkVectorNd& e2, const hkVectorNd& planePoint)
{
	hkVectorNd u;
	u.setSub(p, planePoint);

	const hkSimdDouble64 t0 = u.dot(e1);
	const hkSimdDouble64 t1 = u.dot(e2);

	return u.lengthSquared() - t0 * t0 - t1 * t1;
}

#ifdef HK_DEBUG

/* static */void hkVectorNd::selfTest()
{
	{
		hkVector4d a; a.set(1,2,3,0);
		hkVector4d b; b.set(2,-3,-5,0);
		hkVector4d c; c.set(3,-9,7,0);
		hkVector4d p; p.set(4,6,9,0);

		hkVectorNd an(&a(0), 3);
		hkVectorNd bn(&b(0), 3);
		hkVectorNd cn(&c(0), 3);
		hkVectorNd pn(&p(0), 3);

		hkDouble64 distSq1;
		{
			// Work out the distance of point from the plane
			hkVector4d e0; e0.setSub(b, a);
			hkVector4d e1; e1.setSub(c, a);
			hkVector4d n; n.setCross(e0, e1);

			n.normalize<3>();

			hkVector4d v; v.setSub(p, a);
			distSq1 = v.dot<3>(n).getReal();
			distSq1 *= distSq1;
		}

		// Work it out in n dimensional way
		hkDouble64 distSq2;
		{
			hkVectorNd e1, e2;
			calcUnitAxes(an, bn, cn, e1, e2);

			distSq2 = calcPlaneDistanceSquared(pn, e1, e2, an).getReal();
		}

		HK_ASSERT(0x32432432, hkMath::fabs(distSq1 - distSq2) < 1e-5f);
	}
	

    {
        HK_ALIGN_DOUBLE(const hkDouble64 avals[]) = { 1, 2, 3, 4, 5 };

        hkVectorNd a;

        for (int i = 0; i < int(HK_COUNT_OF(avals)); i++)
        {
            a.set(avals, i);
            HK_ASSERT(0x2432a434, a.isOk());

            HK_ASSERT(0x5443b542, a.equalsStrided(avals, i, 1));
        }
    }

	const hkDouble64 epsilon = 1e-5f;

    const hkDouble64 avals[] = { 1, 2, 3, 4 };
    hkVectorNd a(avals, HK_COUNT_OF(avals));

    hkVector4d av; av.load<4,HK_IO_NATIVE_ALIGNED>(avals);
	HK_ON_DEBUG(hkSimdDouble64 lenDiff; lenDiff.setAbs(av.lengthSquared<4>()- a.lengthSquared()); )
	HK_ASSERT(0x2442323, lenDiff.getReal() < epsilon );

	HK_ON_DEBUG( lenDiff.setAbs(av.length<4>() - a.length()); )
	HK_ASSERT(0x5435334, lenDiff.getReal() < epsilon);

    const hkDouble64 bvals[] = { -1, 4, 2, 3 };
    hkVectorNd b(bvals, HK_COUNT_OF(bvals));

    hkVector4d bv; bv.load<4,HK_IO_NATIVE_ALIGNED>(bvals);

	HK_ON_DEBUG( lenDiff.setAbs(a.dot(b) - av.dot<4>(bv)); )
	HK_ASSERT(0x3634fd00, lenDiff.getReal() < epsilon);

	{
		hkInplaceVectorNd<5> vec;

		vec.setZero();
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
