/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Vector/hkVectorNf.h>

/* static */const hkVector4ComparisonMask::Mask hkVectorNf::s_zeroMask[4] = 
{
	hkVector4ComparisonMask::MASK_XYZW,
	hkVector4ComparisonMask::MASK_X,
	hkVector4ComparisonMask::MASK_XY,
	hkVector4ComparisonMask::MASK_XYZ,
};

hkVectorNf::hkVectorNf(int size):
    m_size(size),
    m_flags(0)
{
    if (size > 0)
    {
        int numEle = (m_size + 3) >> 2;
#if !defined(HK_PLATFORM_PS3_SPU)
        m_elements = hkAlignedAllocate<hkVector4f>(HK_FLOAT_ALIGNMENT, numEle, HK_MEMORY_CLASS_MATH);
#else
		m_elements = hkAllocateChunk<hkVector4f>( numEle, HK_MEMORY_CLASS_MATH);
#endif
		// Zero out last members
        m_elements[numEle - 1].setZero();
        m_flags = IS_ALLOCATED;
    }
}

hkVectorNf::hkVectorNf(const hkFloat32* v, int size):
    m_size(size),
    m_flags(0)
{
    if (size > 0)
    {
        const int numEle = (m_size + 3) >> 2;
#if !defined(HK_PLATFORM_PS3_SPU)
		m_elements = hkAlignedAllocate<hkVector4f>(HK_FLOAT_ALIGNMENT, numEle, HK_MEMORY_CLASS_MATH);
#else
		m_elements = hkAllocateChunk<hkVector4f>( numEle, HK_MEMORY_CLASS_MATH);
#endif
        _setValues(m_elements, v, size);
        m_flags = IS_ALLOCATED;
    }
}

void hkVectorNf::set(const hkFloat32* v, int size)
{
    _setSize(size);
    if (size > 0)
    {
        _setValues(m_elements, v, size);
    }
}

/* static */void hkVectorNf::_setValues(hkVector4f* dst, const hkFloat32* v, int size)
{
    const int numEle = (size + 3) >> 2;

    hkVector4f* end = dst + numEle;

    if ((hk_size_t(v) & (HK_FLOAT_ALIGNMENT - 1)) == 0)
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
    hkVector4f& last = end[-1];

	hkVector4fComparison mask; mask.set(s_zeroMask[size & 3]);
	last.zeroIfFalse(mask);
}

void hkVectorNf::_setSizeAndZeroLast(int size)
{
    _setSize(size);
    if (size & 3)
    {
        const int numEle = (m_size + 3) >> 2;
        hkVector4f& last = m_elements[numEle - 1];

		hkVector4fComparison mask; mask.set(s_zeroMask[size & 3]);
		last.zeroIfFalse(mask);
    }
}

void hkVectorNf::setSizeAndZero(int size)
{
    _setSize(size);
    setZero();
}

void hkVectorNf::setZero()
{
    const int numEle = (m_size + 3) >> 2;

    hkVector4f* dst = m_elements;
    hkVector4f* end = dst + numEle;

    // Zero it
    for (; dst != end; dst++) 
	{
		dst->setZero();
	}
}

void hkVectorNf::negate()
{
	const int numEle = (m_size + 3) >> 2;

	hkVector4f* dst = m_elements;
	hkVector4f* end = dst + numEle;

	// Neg it
	for (; dst != end; dst++) 
	{
		dst->setNeg<4>(*dst);
	}
}

void hkVectorNf::_setSize(int size)
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
				m_elements = hkAlignedAllocate<hkVector4f>(HK_FLOAT_ALIGNMENT, numNewEle, HK_MEMORY_CLASS_MATH);
#else
				m_elements = hkAllocateChunk<hkVector4f>( numNewEle, HK_MEMORY_CLASS_MATH);
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
				hkAlignedDeallocate<hkVector4f>( m_elements );
#else
				hkDeallocateChunk<hkVector4f>( m_elements, numEle, HK_MEMORY_CLASS_MATH);
#endif
				m_flags = 0;
			}
			// Allocate new if need space
			if (numNewEle)
			{
#if !defined(HK_PLATFORM_PS3_SPU)
				m_elements = hkAlignedAllocate<hkVector4f>(HK_FLOAT_ALIGNMENT, numNewEle, HK_MEMORY_CLASS_MATH);
#else
				m_elements = hkAllocateChunk<hkVector4f>( numNewEle, HK_MEMORY_CLASS_MATH);
#endif
				m_flags = IS_ALLOCATED;
			}
		}
    }
    // Set the size
    m_size = size;
}

hkVectorNf::hkVectorNf(const hkVectorNf& rhs):
    m_size(rhs.m_size),
    m_flags(0)
{
    if (rhs.m_size)
    {
        const int numEle = (rhs.m_size + 3) >> 2;

#if !defined(HK_PLATFORM_PS3_SPU)
		m_elements = hkAlignedAllocate<hkVector4f>(HK_FLOAT_ALIGNMENT, numEle, HK_MEMORY_CLASS_MATH);
#else
		m_elements = hkAllocateChunk<hkVector4f>( numEle, HK_MEMORY_CLASS_MATH);
#endif

        hkVector4f* dst = m_elements;
        const hkVector4f* src = rhs.m_elements;
        const hkVector4f* end = dst + numEle;
        // Copy
        while (dst != end) *dst++ = *src++;

        m_flags = IS_ALLOCATED;
    }
}

void hkVectorNf::operator=(const hkVectorNf& rhs)
{
    _setSize(rhs.m_size);

    // Copy
    {
        const int numEle = (rhs.m_size + 3) >> 2;
        hkVector4f* dst = m_elements;
        const hkVector4f* src = rhs.m_elements;
        const hkVector4f* end = dst + numEle;

        while (dst != end) *dst++ = *src++;
    }
}

hkSimdFloat32 hkVectorNf::lengthSquared() const
{
    HK_ASSERT(0x23432432, m_size > 0);

    int numEle = (m_size + 3) >> 2;

    const hkVector4f* src = m_elements;
    const hkVector4f* end = src + numEle;

    hkVector4f total;
    total.setMul(*src, *src);
    src++;

    for (; src != end; src++)
    {
        total.addMul(*src, *src);
    }
    return total.horizontalAdd<4>();
}

hkSimdFloat32 hkVectorNf::length() const
{
	return lengthSquared().sqrt();
}


hkSimdFloat32 hkVectorNf::dot(const hkVectorNf& rhs) const
{
    HK_ASSERT(0x4234a324, m_size > 0 && m_size == rhs.m_size);

    int numEle = (m_size + 3) >> 2;

    const hkVector4f* a = m_elements;
    const hkVector4f* b = rhs.m_elements;

    hkVector4f total;

    total.setMul(a[0], b[0]);
    for (int i = 1; i < numEle; i++)
    {
        total.addMul(a[i], b[i]);
    }

    return total.horizontalAdd<4>();
}

hkSimdFloat32 hkVectorNf::horizontalAdd() const
{
    hkVector4f total; total.setZero();

    int numEle = (m_size + 3) >> 2;
    const hkVector4f* a = m_elements;
    for (int i = 0; i < numEle; i++)
    {
        total.add(a[i]);
    }
    return total.horizontalAdd<4>();
}

hkBool hkVectorNf::equals(const hkVectorNf& rhs, hkFloat32 epsilon) const
{
    HK_ASSERT(0x2432a432, m_size > 0 && m_size == rhs.m_size);
    if (m_size <= 0 || m_size != rhs.m_size)
    {
        return false;
    }

    int numEle = (m_size + 3) >> 2;

    const hkVector4f* a = m_elements;
    const hkVector4f* b = rhs.m_elements;

	hkSimdFloat32 sEps; sEps.setFromFloat(epsilon);
    for (int i = 0; i < numEle; i++)
    {
        if (!a[i].allEqual<4>(b[i], sEps))
        {
            return false;
        }
    }
    return true;
}

hkBool hkVectorNf::equalsStrided(const hkFloat32* ele, int size, int stride, hkFloat32 epsilon) const
{
    if (size != m_size)
    {
        return false;
    }

    const hkFloat32* a = (const hkFloat32*)m_elements;
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

hkBool hkVectorNf::isOk() const
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
        hkVector4f a;
        const hkVector4f& last = m_elements[numEle - 1];
		
		hkVector4fComparison mask; mask.set(s_zeroMask[m_size & 3]);

		a.setSelect(mask, last, hkVector4f::getZero());

        // All should be exactly equal
		hkSimdFloat32 eps; eps.setFromFloat(1e-3f);
        return a.allEqual<4>(last,eps) != hkFalse32;
    }

    return true;
}

void hkVectorNf::alias(hkVector4f* vecs, int size)
{
    // If it has allocated memory, then free that memory
    if (m_flags & IS_ALLOCATED)
    {
#if !defined(HK_PLATFORM_PS3_SPU)
		hkAlignedDeallocate<hkVector4f>( m_elements );
#else
		hkDeallocateChunk<hkVector4f>( m_elements, getNumVectors(), HK_MEMORY_CLASS_MATH);
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

void hkVectorNf::unalias()
{
    if ((m_flags & IS_ALLOCATED) == 0 && m_size > 0)
    {
        const int numEle = (m_size + 3) >> 2;

        const hkVector4f* src = m_elements;
#if !defined(HK_PLATFORM_PS3_SPU)
        hkVector4f* dst = hkAlignedAllocate<hkVector4f>(HK_FLOAT_ALIGNMENT, numEle, HK_MEMORY_CLASS_MATH);
#else
		hkVector4f* dst = hkAllocateChunk<hkVector4f>( numEle, HK_MEMORY_CLASS_MATH);
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

void hkVectorNf::add(const hkVectorNf& a)
{
	HK_ASSERT(0x5435a435, getSize() > 0 && a.getSize() == getSize());

	const hkVector4f* avec = a.m_elements;
	hkVector4f* dst = m_elements;
	hkVector4f* end = m_elements + ((m_size + 3) >> 2);

	for (; dst < end; dst++, avec++)
	{
		dst->add(*avec);
	}
}

void hkVectorNf::sub(const hkVectorNf& a)
{
	HK_ASSERT(0x5435a435, getSize() > 0 && a.getSize() == getSize());

	const hkVector4f* avec = a.m_elements;
	hkVector4f* dst = m_elements;
	hkVector4f* end = m_elements + ((m_size + 3) >> 2);

	for (; dst < end; dst++, avec++)
	{
		dst->sub(*avec);
	}
}

void hkVectorNf::setSub(const hkVectorNf& a, const hkVectorNf& b)
{
	const int size = a.getSize();
	HK_ASSERT(0x5435a435, size > 0 && b.getSize() == size);
	setSize(size);

	const hkVector4f* avec = a.m_elements;
	const hkVector4f* bvec = b.m_elements;
	hkVector4f* dst = m_elements;
	hkVector4f* end = m_elements + ((m_size + 3) >> 2);

	for (; dst < end; dst++, avec++, bvec++)
	{
		dst->setSub(*avec, *bvec);
	}
}

void hkVectorNf::setAdd(const hkVectorNf& a, const hkVectorNf& b)
{
	const int size = a.getSize();
	HK_ASSERT(0x5435a435, size > 0 && b.getSize() == size);
	setSize(size);

	const hkVector4f* avec = a.m_elements;
	const hkVector4f* bvec = b.m_elements;
	hkVector4f* dst = m_elements;
	hkVector4f* end = m_elements + ((m_size + 3) >> 2);

	for (; dst < end; dst++, avec++, bvec++)
	{
		dst->setAdd(*avec, *bvec);
	}
}

void hkVectorNf::mul(hkSimdFloat32Parameter vIn)
{
	hkVector4f* dst = m_elements;
	hkVector4f* end = m_elements + ((m_size + 3) >> 2);

	for (; dst < end; dst++)
	{
		dst->mul(vIn);
	}
}

void hkVectorNf::setMul(hkSimdFloat32Parameter s, const hkVectorNf& v)
{
	setSize(v.m_size);

	hkVector4f* dst = m_elements;
	hkVector4f* end = m_elements + ((m_size + 3) >> 2);
	hkVector4f* a = v.m_elements;

	for (; dst < end; dst++, a++)
	{
		dst->setMul(*a, s);
	}
}

hkSimdFloat32 hkVectorNf::normalize()
{
	HK_ASSERT(0x32432432, m_size > 0);
	const hkSimdFloat32 len = length();
	if (len.getReal() > 1e-10f)
	{
		hkSimdFloat32 invLen; invLen.setReciprocal(len);
		mul(invLen);
	}
	else
	{
		const int numVecs = (m_size + 3) >> 2;
		m_elements->set(1, 0, 0, 0);

		hkVector4f* dst = m_elements + 1;
		hkVector4f* end = m_elements + numVecs;
		for (; dst < end; dst++)
		{
			dst->setZero();
		}
	}
	return len;
}

/* static */void hkVectorNf::calcUnitAxes(const hkVectorNf& a, const hkVectorNf& b, const hkVectorNf& c, hkVectorNf& e1, hkVectorNf& e2)
{
	HK_ON_DEBUG( const int size = a.getSize(); )
	HK_ASSERT(0x452453a4, size > 0 && size == b.getSize() && size == c.getSize());

	e1.setSub(b, a);
	e1.normalize();

	e2.setSub(c, a);
	
	hkVectorNf t;
	t.setMul(e1.dot(e2), e1);

	e2.sub(t);
	e2.normalize();
}

/* static */hkSimdFloat32 hkVectorNf::calcPlaneDistanceSquared(const hkVectorNf& p, hkVectorNf& e1, hkVectorNf& e2, const hkVectorNf& planePoint)
{
	hkVectorNf u;
	u.setSub(p, planePoint);

	const hkSimdFloat32 t0 = u.dot(e1);
	const hkSimdFloat32 t1 = u.dot(e2);

	return u.lengthSquared() - t0 * t0 - t1 * t1;
}

#ifdef HK_DEBUG

/* static */void hkVectorNf::selfTest()
{
	{
		hkVector4f a; a.set(1,2,3,0);
		hkVector4f b; b.set(2,-3,-5,0);
		hkVector4f c; c.set(3,-9,7,0);
		hkVector4f p; p.set(4,6,9,0);

		hkVectorNf an(&a(0), 3);
		hkVectorNf bn(&b(0), 3);
		hkVectorNf cn(&c(0), 3);
		hkVectorNf pn(&p(0), 3);

		hkFloat32 distSq1;
		{
			// Work out the distance of point from the plane
			hkVector4f e0; e0.setSub(b, a);
			hkVector4f e1; e1.setSub(c, a);
			hkVector4f n; n.setCross(e0, e1);

			n.normalize<3>();

			hkVector4f v; v.setSub(p, a);
			distSq1 = v.dot<3>(n).getReal();
			distSq1 *= distSq1;
		}

		// Work it out in n dimensional way
		hkFloat32 distSq2;
		{
			hkVectorNf e1, e2;
			calcUnitAxes(an, bn, cn, e1, e2);

			distSq2 = calcPlaneDistanceSquared(pn, e1, e2, an).getReal();
		}

		HK_ASSERT(0x32432432, hkMath::fabs(distSq1 - distSq2) < 1e-5f);
	}
	

    {
        HK_ALIGN_FLOAT(const hkFloat32 avals[]) = { 1, 2, 3, 4, 5 };

        hkVectorNf a;

        for (int i = 0; i < int(HK_COUNT_OF(avals)); i++)
        {
            a.set(avals, i);
            HK_ASSERT(0x2432a434, a.isOk());

            HK_ASSERT(0x5443b542, a.equalsStrided(avals, i, 1));
        }
    }

	const hkFloat32 epsilon = 1e-5f;

    const hkFloat32 avals[] = { 1, 2, 3, 4 };
    hkVectorNf a(avals, HK_COUNT_OF(avals));

    hkVector4f av; av.load<4,HK_IO_NATIVE_ALIGNED>(avals);
	HK_ON_DEBUG(hkSimdFloat32 lenDiff; lenDiff.setAbs(av.lengthSquared<4>()- a.lengthSquared()); )
	HK_ASSERT(0x2442323, lenDiff.getReal() < epsilon );

	HK_ON_DEBUG( lenDiff.setAbs(av.length<4>() - a.length()); )
	HK_ASSERT(0x5435334, lenDiff.getReal() < epsilon);

    const hkFloat32 bvals[] = { -1, 4, 2, 3 };
    hkVectorNf b(bvals, HK_COUNT_OF(bvals));

    hkVector4f bv; bv.load<4,HK_IO_NATIVE_ALIGNED>(bvals);

	HK_ON_DEBUG( lenDiff.setAbs(a.dot(b) - av.dot<4>(bv)); )
	HK_ASSERT(0x3634fd00, lenDiff.getReal() < epsilon);

	{
		hkInplaceVectorNf<5> vec;

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
