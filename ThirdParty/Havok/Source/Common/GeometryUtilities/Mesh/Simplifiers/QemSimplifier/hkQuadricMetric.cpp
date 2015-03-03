/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>


#include <Common/Base/Algorithm/Sort/hkSort.h>

// this
#include <Common/GeometryUtilities/Mesh/Simplifiers/QemSimplifier/hkQuadricMetric.h>

void hkQuadricMetric::calcA(const hkVectorN& e1, const hkVectorN& e2, hkMatrixNm& a)
{
	// I - e1e1^t - e2e2^t

#if 1
	HK_ASSERT(0x3424a324, e1.getSize() == e2.getSize());
	const int size = e1.getSize();

	hkMatrixNm e1o, e2o;

	e1o.setOuterProduct(e1, e1);
	e2o.setOuterProduct(e2, e2);

	a.setSize(size, size);
	a.setIdentity();
	a.sub(e1o);
	a.sub(e2o);
#else
	HK_ASSERT(0x3424a324, e1.getSize() == e2.getSize());

	const int size = e1.getSize();

	a.setSize(size, size);

	const hkVector4*const e1vecs = e1.getVectors();
	const hkVector4*const e2vecs = e2.getVectors();
	hkVector4* dst = a.getVectors();

	const int numVecs = (size + 3) >> 2;

	hkVector4 ones; ones.setAll(1);

	for (int i = 0; i < size; i++)
	{
		hkVector4 v1; v1.setBroadcast(i & 3, e1vecs[i >> 2]);
		v1.setNeg<4>(v1);

		hkVector4 v2; v2.setBroadcast(i & 3, e2vecs[i >> 2]);
		v2.setNeg<4>(v2);

		const hkVector4* e1cur = e1vecs;
		const hkVector4* e2cur = e2vecs;

		const int colVec = i >> 2;

		for (int j = 0; j < numVecs; j++, e1cur++, e2cur, dst++)
		{
			if (j == colVec)
			{
				// Set the identity part
				hkVector4 t = ones;
				t.zeroElement(i & 3);
				dst->setSub4(ones, t);

				dst->addMul4(*e1cur, v1);
				dst->addMul4(*e2cur, v2);
			}
			else
			{
				// Work out the outer product
				dst->setMul4(*e1cur, v1);
				dst->addMul4(*e2cur, v2);
			}
		}
	}
#endif

}

hkSimdReal hkQuadricMetric::calcDistanceSquared(const hkVectorN& a) const
{
	// d^2 = v^tAv + 2b^tv + c
	hkInplaceVectorN<16> t;
	m_a.multiply(a, t);
	const hkSimdReal mc = hkSimdReal::fromFloat(m_c);
	return t.dot(a) + hkSimdReal_2 * m_b.dot(a) + mc;
}

hkSimdReal hkQuadricMetric::calcDistanceSquared(hkVector4Parameter aIn) const
{
	hkVector4 aV; aV.setXYZ_0(aIn);

	// d^2 = v^tAv + 2b^tv + c
	hkVectorN a; a.alias(&aV, 3);

	hkInplaceVectorN<16> t;
	m_a.multiply(a, t);
	const hkSimdReal mc = hkSimdReal::fromFloat(m_c);
	return t.dot(a) + hkSimdReal_2 * m_b.dot(a) + mc;
}


void hkQuadricMetric::setFromPlane(const hkVectorN& a, const hkVectorN& b, const hkVectorN& c)
{
	hkInplaceVectorN<16> e1, e2;
	hkVectorN::calcUnitAxes(a, b, c, e1, e2);

	calcA(e1, e2, m_a);

	hkInplaceVectorN<16> t; t.setMul(a.dot(e2), e2);
	m_b.setMul(a.dot(e1), e1);
	m_b.add(t);
	m_b.sub(a);

	const hkSimdReal t0 = a.dot(e1);
	const hkSimdReal t1 = a.dot(e2);
	(a.dot(a) - t0 * t0 - t1 * t1).store<1>(&m_c);
}


hkBool hkQuadricMetric::isOk() const
{
	return 
		m_a.isOk() && 
		m_b.isOk() && 
		m_a.isSymmetric() && 
		m_a.getNumColumns() == m_b.getSize();
}

int hkQuadricMetric::getStoreSize() const 
{ 
	HK_ASSERT(0x23443234, isOk());
	return calcStoreSize(m_b.getSize());
}
	
void hkQuadricMetric::store(hkReal* dataIn) const
{
	hkReal* data = dataIn;
	// Store the symmetric matrix
	const int stride = m_a.getRealStride();
	const hkReal* t = m_a.getRealElements();
	const int size = m_b.getSize();
	
	for (int i = 0; i < size; i++)
	{
		const hkReal* c = t;
		for (int j = i; j < size; j++)
		{
			*data++ = *c++;
		}
		t += stride + 1;
	}

	// Output b
	t = m_b.getElements();
	for (int i = 0; i < size; i++)
	{
		*data++ = *t++;
	}
	// Finally output c
	*data++ = m_c;

	HK_ASSERT(0x3242a343, data - dataIn == calcStoreSize(size));
}

void hkQuadricMetric::load(int size, const hkReal* data)
{
	m_a.setSize(size, size);
	m_b.setSize(size);

	const int stride = m_a.getRealStride();
	hkReal* t = m_a.getRealElements();
	
	for (int i = 0; i < size; i++)
	{
		*t = *data++;

		hkReal* c1 = t + 1;
		hkReal* c2 = t + stride;
		
		for (int j = i + 1; j < size; j++)
		{
			const hkReal v = *data++;
			*c1 = v;
			*c2 = v;

			c1 ++;
			c2 += stride;
		}
		t += stride + 1;
	}

	HK_ASSERT(0x24343a24, m_a.isSymmetric());

	// Copy the vector
	t = m_b.getElements();

	for (int i = 0; i < size; i++)
	{
		*t++ = *data++;
	}
	m_c = *data++;
}

hkBool hkQuadricMetric::equals(const hkQuadricMetric& rhs, hkReal threshold)
{
	if (this == &rhs) return true;

	return m_a.equals(rhs.m_a, threshold) &&
		   m_b.equals(rhs.m_b, threshold) &&
		   hkMath::fabs(m_c - rhs.m_c) < threshold;
}

void hkQuadricMetric::setFromPlane(const hkReal* values, int size)
{
	hkInplaceVectorN<16> a;
	hkInplaceVectorN<16> b;
	hkInplaceVectorN<16> c;

	a.set(values, size);
	b.set(values + size, size);
	c.set(values + size * 2, size);

	setFromPlane(a, b, c);
}


#ifdef HK_DEBUG

/* static */void hkQuadricMetric::selfTest()
{
	{
		hkVector4 a; a.set( 1, 2, 3);
		hkVector4 b; b.set( 2, -3, -5);
		hkVector4 c; c.set( 3, -9, 7);

		hkVectorN an(&a(0), 3);
		hkVectorN bn(&b(0), 3);
		hkVectorN cn(&c(0), 3);

		hkQuadricMetric qm;
		qm.setFromPlane(an, bn, cn);

		const hkSimdReal epsilon = hkSimdReal::fromFloat(1e-5f);

		HK_ON_DEBUG(hkSimdReal distSq; distSq.setAbs(qm.calcDistanceSquared(an)); )
		HK_ASSERT(0x2423432, distSq < epsilon);
		HK_ON_DEBUG(distSq.setAbs(qm.calcDistanceSquared(bn)); )
		HK_ASSERT(0x2423432, distSq < epsilon);
		HK_ON_DEBUG(distSq.setAbs(qm.calcDistanceSquared(cn)); )
		HK_ASSERT(0x2423432, distSq < epsilon);

		HK_ASSERT(0x2423423, qm.isOk());
		HK_ASSERT(0x2423432, qm.equals(qm));

		hkQuadricMetric qm2(qm);
		HK_ASSERT(0x2423432, qm.equals(qm2));

		hkQuadricMetric qm3;
		HK_ASSERT(0x2423432, !qm.equals(qm3));

		hkArray<hkReal> store;
		store.setSize(qm.getStoreSize());
		qm.store(store.begin());

		qm3.load(qm.getSize(), store.begin());

		HK_ASSERT(0x32423432, qm.equals(qm3));

		//hkVector4 p(4, 6, 9);
		//hkVectorN pn(&p(0), 3);
	}
	{
		hkQuadricMetric qm1;
		{
			const hkReal v[] = 
			{
				1, 3, 2,
				2, 3, 2,
				2.5, 3, 3,
			};
			qm1.setFromPlane(v, 3);
		}
		hkQuadricMetric qm2;
		{
			const hkReal v[] = 
			{
				3, 1, -2,
				-2, 1, -2,
				1, 2, -2,
			};
			qm2.setFromPlane(v, 3);
		}

		HK_ASSERT(0x234324, qm1.getStoreSize() == 10);
		hkVector4 t0[3];
		hkVector4 t1[3];
		hkVector4 r[3];

		qm1.store((hkReal*)t0);
		qm2.store((hkReal*)t1);

		for (int i = 0; i < 3; i++)
		{
			r[i].setAdd(t0[i], t1[i]);
		}

		hkQuadricMetric qm; 
		qm.load(3, (hkReal*)r);

#if 0
		// 
		hkVector4 p0; p0.set(0, 3, -2);
		hkReal dist0 = qm.calcDistanceSquared(p0);

		hkVector4 p1; p1.set(3, 3, -2);
		hkReal dist1 = qm.calcDistanceSquared(p1);


		hkVector4 p2; p2.set(3, 3, 0);
		hkReal dist2 = qm.calcDistanceSquared(p2);


		// 5? or 25?
		hkVector4 p3; p3.set(3, 2, 0);
		hkReal dist3 = qm.calcDistanceSquared(p3);
#endif
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
