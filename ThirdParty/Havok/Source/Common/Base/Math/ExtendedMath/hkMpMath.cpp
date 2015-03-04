/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/ExtendedMath/hkMpMath.h>

//
// hkMpUint
//

//
int		compare(const hkMpUint& bi0, const hkMpUint& bi1)
{
	if(&bi0 == &bi1)	return 0;
		
	const int szDelta = bi0.getSize() - bi1.getSize();
	if(szDelta < 0)	return -1;
	if(szDelta > 0)	return +1;
		
	for(int i = bi0.getSize() - 1; i >= 0; --i)
	{
		if(bi0[i] < bi1[i])	return -1;
		if(bi0[i] > bi1[i])	return +1;
	}
		
	return 0;
}

//
void	pow(hkMpUint& bi, unsigned exp)
{
	if(exp == 0)
	{
		set(bi,1);
	}
	else if(exp > 1)
	{
		hkMpUint square = bi;
		hkMpUint value; set(value,1);
		do
		{
			if(exp & 1)
			{
				mul(value, square, value);
			}
			mul(square, square, square);
			exp >>= 1;
		} while(exp);
		bi = value;
	}
}

//
void	add(const hkMpUint& bi0, const hkMpUint& bi1, hkMpUint& biOut)
{
	hkMpUint	biTmp;
	const bool	aliased = &biOut == &bi0 || &biOut == &bi1;
	hkMpUint&	output = aliased ? biTmp : biOut;

	if(bi0.isZero())
	{
		output = bi0;
	}
	else if(bi1.isZero())
	{
		output = bi1;
	}
	else
	{
		const hkMpUint&	opA = bi0.getSize() > bi1.getSize() ? bi0 : bi1;
		const hkMpUint&	opB = bi0.getSize() > bi1.getSize() ? bi1 : bi0;
		bool				carry = false;
		int					index = 0;
			
		output.m_atoms.setSize(opA.getSize() + 1);

		for(; index < opB.getSize(); ++index)
		{
			hkMpUint::Atom	value = opA[index] + opB[index];
			bool			overflow = value < opA[index];
			if(carry)
			{
				++value;
				overflow |= (value == 0);
			}
			output[index]	=	value;
			carry			=	overflow;
		}

		for(; carry && index < opA.getSize(); ++index)
		{
			const hkMpUint::Atom value = opA[index] + 1;
			carry = (value == 0);
			output[index] = value;
		}

		for(; index < opA.getSize(); ++index)
		{
			output[index] = opA[index];
		}

		if(carry) output[index] = 1; else output.m_atoms.popBack();
	}

	if(aliased)
	{
		biOut = output;
	}
}

//
void	sub(const hkMpUint& bi0, const hkMpUint& bi1, hkMpUint& biOut)
{
	hkMpUint	biTmp;
	const bool	aliased = &biOut == &bi0 || &biOut == &bi1;
	hkMpUint&	output = aliased ? biTmp : biOut;

	if(bi1.isZero())
	{
		output = bi0;
	}
	else if(bi0.getSize() < bi1.getSize()) 
	{
		HK_ERROR(0x68707453, "Underflow");
	}
	else
	{
		output.m_atoms.setSize(bi0.getSize());

		int		index = 0;
		bool	carry = false;

		for(; index < bi1.getSize(); ++index)
		{
			hkMpUint::Atom	value = bi0[index] - bi1[index];
			bool			underflow = value > bi0[index];
			if(carry)
			{
				underflow |= (value == 0);
				--value;
			}

			output[index]	=	value;
			carry			=	underflow;
		}

		for(; carry && index < bi0.getSize(); ++index)
		{
			carry			=	(bi0[index] == 0);
			output[index]	=	bi0[index] - 1;
		}

		if(carry)
		{
			HK_ERROR(0x68707453, "Underflow");
		}
		else
		{
			for(; index < bi0.getSize(); ++index)
			{
				output[index]	=	bi0[index];
			}
			output.stripLeadingZeros();
		}
	}

	if(aliased) biOut = output;
}

//
void		mul(const hkMpUint& bi0, const hkMpUint& bi1, hkMpUint& biOut)
{
	hkMpUint	biTmp;
	const bool	aliased = &biOut == &bi0 || &biOut == &bi1;
	hkMpUint&	output = aliased ? biTmp : biOut;

	if(bi0.isZero() || bi1.isZero()) output.setZero();
	else
	{
		output.m_atoms.setSize(bi0.getSize() + bi1.getSize());
		for(int i=0; i<output.getSize(); ++i) output[i] = 0;

		for(int index0=0; index0 < bi0.getSize(); ++index0)
		{
			const hkMpUint::Atom	v0 = bi0[index0];
			bool					carry = false;
			for(int bit = 0; bit < hkMpUint::BITS_PER_ATOM; ++bit)
			{
				if(0 == (hkMpUint::Atom(1) & (v0 >> bit))) continue;
				int index2 = index0;
				for(int index1 = 0; index1 <= bi1.getSize(); ++index1, ++index2)
				{
					hkMpUint::Atom	value = output[index2] + bi1.getAtom(index1, bit);
					bool			overflow = value < output[index2];
					if(carry)
					{
						value++;
						carry |= (value == 0);
					}
					output[index2] = value;
					carry = overflow;
				}
				for(; carry; ++index2)
				{
					output[index2]++;
					carry = (output[index2] == 0);
				}
			}
		}

		output.stripLeadingZeros();
	}

	if(aliased)
	{
		biOut = output;
	}
}

//
void	div(const hkMpUint& bi0, const hkMpUint& bi1, hkMpUint& qOut)
{
	hkMpUint r;
	div(bi0,bi1,qOut,r);
}

//
void	mod(const hkMpUint& bi0, const hkMpUint& bi1, hkMpUint& rOut)
{
	hkMpUint q;
	div(bi0,bi1,q,rOut);
}

//
void	div(const hkMpUint& bi0, const hkMpUint& bi1, hkMpUint& qOut, hkMpUint& rOut)
{
	hkMpUint	biTmpQ,biTmpR;
	const bool	aliasedQ = &qOut == &bi0 || &qOut == &bi1;
	hkMpUint&	outputQ = aliasedQ ? biTmpQ : qOut;
	const bool	aliasedR = &rOut == &bi0 || &rOut == &bi1;
	hkMpUint&	outputR = aliasedR ? biTmpR : rOut;

	outputQ.setZero();
	outputR.setZero();
		
	if(bi0.getSize() >= bi1.getSize())
	{
		if(bi1.isZero())
		{
			HK_ERROR(0x21B66586, "Division by zero");
		}
		else
		{
			const int baseSize = bi0.getSize();
				
			outputR = bi0;
			outputR.m_atoms.pushBack(0);
				
			hkMpUint::AtomArray buffer; buffer.setSize(outputR.getSize(),0);

			outputQ.m_atoms.setSize(baseSize - bi1.getSize() + 1);
			for(int i=0; i<outputQ.getSize(); ++i) outputQ[i] = 0;

			for(int index0 = outputQ.getSize()-1; index0 >= 0; --index0)
			{
				outputQ[index0] = 0;
				for(int bit = hkMpUint::BITS_PER_ATOM-1; bit >= 0; --bit)
				{
					int index2 = index0;
					bool carry = false;
					for(int index1 = 0; index1 <= bi1.getSize(); ++index1, ++index2)
					{
						hkMpUint::Atom	value = outputR[index2] - bi1.getAtom(index1, bit);
						bool			underflow = value > outputR[index2];
						if(carry)
						{
							underflow |= (value == 0);
							value--;
						}
						buffer[index2] = value;
						carry = underflow;
					}

					for(; carry && index2 < baseSize; ++index2)
					{
						carry = (outputR[index2] == 0);
						buffer[index2] = outputR[index2] - 1;
					}

					if(!carry)
					{
						outputQ[index0] |= (hkMpUint::Atom(1) << bit);
						while(index2-- > index0)
						{
							outputR[index2] = buffer[index2];
						}
					}
				}					
			}

			outputQ.stripLeadingZeros();
			outputR.stripLeadingZeros();
		}
	}
	else
	{
		outputR = bi0;
	}

	if(aliasedQ) qOut = outputQ;
	if(aliasedR) rOut = outputR;
}

//
void	gcd(const hkMpUint& x, const hkMpUint& y, hkMpUint& output)
{
	hkMpUint tx = x;
	hkMpUint ty = y;
	hkMpUint q,r;
	while(!ty.isZero())
	{		
		mod(tx,ty,r);
		tx = ty;
		ty = r;
	}
	output = tx;
}

//
void	shift(hkMpUint& x, int count)
{
	if(!x.isZero() && count)
	{
		if(count > 0)
		{
			// left shift (x << count)
			const int numExtraAtoms = 1 + count / hkMpUint::BITS_PER_ATOM;
			const int numAtoms = numExtraAtoms - 1;
			const int psz = x.getSize() - 1;		
			const int csz = psz + numAtoms;
			x.m_atoms.expandBy(numExtraAtoms);		
			count -= numAtoms * hkMpUint::BITS_PER_ATOM;		
			for(int i = 0; i <= psz; ++i) x.m_atoms[csz - i] = x.m_atoms[psz - i];
			for(int i = 0; i < numAtoms; ++i) x.m_atoms[i] = 0;
			x.m_atoms.back() = 0;
			while(count)
			{
				const int numBits = hkMath::min2(count,1);
				const int shifts = hkMpUint::BITS_PER_ATOM - numBits;
				count -= numBits;
				for(int i = x.getSize()-1; i > numAtoms; --i)
				{
					x.m_atoms[i]	<<=	numBits;
					x.m_atoms[i]	|=	x.m_atoms[i-1] >> shifts;
				}
				x.m_atoms[numAtoms] <<= numBits;
			}
			x.stripLeadingZeros();
		}
		else
		{
			// right shift (x >> count)
			HK_ERROR(0x150166ae, "Not implemented");
		}
	}
}

//
void	toString(const hkMpUint& bi, unsigned base, hkStringBuf& stringOut)
{
	HK_ASSERT2(0x91AAE7D2, base <= 36, "Base out-of-range.");
		
	static const char*		digits = "0123456789abcdefghijklmnopqrstuvwxyz";
	hkInplaceArray<char,32>	str;
	if(bi.isZero())
	{
		str.pushBack('0');
	}
	else
	{
		hkMpUint	biBase; set(biBase, base);
		hkMpUint	biValue = bi;
		hkMpUint	biRemainder;
		while(!biValue.isZero())
		{
			div(biValue, biBase, biValue, biRemainder);
			str.pushBack(digits[biRemainder.isZero() ? 0 : biRemainder[0]]);
		}

		for(int i = 0; i < (str.getSize() >> 1); ++i)
		{
			const char t = str[i];
			str[i]					=	str[str.getSize()-i-1];
			str[str.getSize()-i-1]	=	t;
		}
	}

	stringOut.set(str.begin(),str.getSize());
}

//
// hkMpRational
//

//
int	compare(const hkMpRational& rat0, const hkMpRational& rat1)
{
	if(rat0.m_signed && !rat1.m_signed) return -1;
	if(!rat0.m_signed && rat1.m_signed) return +1;
	
	hkMpUint	p0; mul(rat0.m_num, rat1.m_den, p0);
	hkMpUint	p1; mul(rat0.m_den, rat1.m_num, p1);
	return compare(p0,p1) * (rat0.m_signed ? -1 : +1);
}

//
int compare(const hkMpRational& rat, const hkMpUint& bi)
{
	if(rat.m_signed) return -1;

	const bool z0 = rat.isZero();
	const bool z1 = bi.isZero();
	if(z0)
	{
		if(z1)
		{
			return 0;
		}
		else
		{
			return -1;
		}
	}
	else
	{
		if(z1)
		{
			return 1;
		}
		else
		{
			hkMpUint	p; mul(rat.m_den, bi, p);
			return compare(rat.m_num,p) * (rat.m_signed ? -1 : +1);
		}
	}
}

//
void inv(hkMpRational& rat)
{
	const hkMpUint t = rat.m_num;
	rat.m_num = rat.m_den;
	rat.m_den = t;
}
	
//
void pow(hkMpRational& rat, int exp)
{
	const bool zero = rat.isZero();
	const bool negexp = exp < 0;
	if(zero)
	{
		if(exp == 0)
		{
			HK_ERROR(0x06DCD567, "Undefined.");
		}
		if(negexp)
		{
			HK_ERROR(0x06DCD567, "Zero to a negative exponent.");
		}
		rat.setNumeratorAndDemominator(0, 1);
		return;
	}

	if(exp == 0) { rat.setNumeratorAndDemominator(1,1); return; }
	if(exp == 1) return;
	if(exp == -1) { inv(rat); return; }

	const unsigned		c = unsigned(negexp ? -exp : exp);
	pow(rat.m_num, c);
	pow(rat.m_den, c);
	
	if(negexp)
	{
		inv(rat);
	}
}

//
void compact(hkMpRational& rat)
{
	hkMpUint cd; gcd(rat.m_num, rat.m_den, cd);
	if(!cd.isZero())
	{
		hkMpUint r;
		div(rat.m_num, cd, rat.m_num, r);
		div(rat.m_den, cd, rat.m_den, r);
	}
	if(rat.isZero()) rat.m_signed = false;
}

//
void sum(const hkMpRational& rat0, bool sgn0, const hkMpRational& rat1, bool sgn1, hkMpRational& ratOut)
{
	if(rat0.isZero()) { ratOut = rat1; ratOut.m_signed = sgn1; ratOut.cleanZero(); return; }
	if(rat1.isZero()) { ratOut = rat0; ratOut.m_signed = sgn0; ratOut.cleanZero(); return; }

	hkMpUint	ab; mul(rat0.m_num, rat1.m_den, ab);
	hkMpUint	bc; mul(rat0.m_den, rat1.m_num, bc);
	hkMpUint	bd; mul(rat0.m_den, rat1.m_den, bd);
	const int	cmp = compare(ab,bc);
	const int	type =	(cmp <= 0 ? 1 : 0) | (sgn0 ? 2 : 0) | (sgn1 ? 4 : 0);
	switch(type)
	{
		case	0:	add(ab, bc, ratOut.m_num), ratOut.m_signed = false; break;
		case	1:	add(ab, bc, ratOut.m_num), ratOut.m_signed = false; break;
		case	2:	sub(ab, bc, ratOut.m_num), ratOut.m_signed = true; break;
		case	3:	sub(bc, ab, ratOut.m_num), ratOut.m_signed = false; break;
		case	4:	sub(ab, bc, ratOut.m_num), ratOut.m_signed = false; break;
		case	5:	sub(bc, ab, ratOut.m_num), ratOut.m_signed = true; break;
		case	6:	add(ab, bc, ratOut.m_num), ratOut.m_signed = true; break;
		case	7:	add(ab, bc, ratOut.m_num), ratOut.m_signed = true; break;
	}
	ratOut.m_den		=	bd;
	ratOut.cleanZero();
}
	
//
void add(const hkMpRational& rat0, const hkMpRational& rat1, hkMpRational& ratOut)
{
	sum(rat0, rat0.m_signed, rat1, rat1.m_signed, ratOut);
}

//
void sub(const hkMpRational& rat0, const hkMpRational& rat1, hkMpRational& ratOut)
{
	sum(rat0, rat0.m_signed, rat1, !rat1.m_signed, ratOut);
}

//
void mul(const hkMpRational& rat0, const hkMpRational& rat1, hkMpRational& ratOut)
{
	hkMpUint n,d;
	mul(rat0.m_num, rat1.m_num, n);
	mul(rat0.m_den, rat1.m_den, d);
	ratOut.m_num = n;
	ratOut.m_den = d;
	ratOut.m_signed = (rat0.getSign() * rat1.getSign()) < 0;
}

//
void mul(const hkMpRational& rat, const hkMpUint& bi, hkMpRational& ratOut)
{
	mul(rat.m_num, bi, ratOut.m_num);
	ratOut.m_signed = rat.m_signed;
}

//
void div(const hkMpRational& rat0, const hkMpRational& rat1, hkMpRational& ratOut)
{
	hkMpUint n,d;
	mul(rat0.m_num, rat1.m_den, n);
	mul(rat0.m_den, rat1.m_num, d);
	ratOut.m_num = n;
	ratOut.m_den = d;
	ratOut.m_signed = (rat0.getSign() * rat1.getSign()) < 0;
}

//
void toString(const hkMpRational& rat, unsigned base, hkStringBuf& stringOut)
{
	hkStringBuf sn; toString(rat.m_num,base,sn);
	hkStringBuf sd; toString(rat.m_den,base,sd);
	stringOut.printf("%s%s/%s",rat.m_signed?"-":"",sn.cString(),sd.cString());
}

//
void get(const hkMpRational& rat, hkMpUint& valueOut)
{
	div(rat.m_num, rat.m_den, valueOut);
}

//
// Unit tests.
//

#ifdef HK_PLATFORM_WIN32

#include <Common/Internal/GeometryProcessing/hkGeometryProcessing.h>

//
static int	unitTestMpRationalBasics(int iterations)
{
	hkGeometryProcessing::Prng	prng;
	const int					range = 32768;
	for(int i=0; i<iterations; ++i)
	{
		int	i0		= (prng.nextUint32() % range) - (range >> 1);
		int	i1		= (prng.nextUint32() % range) - (range >> 1);
		
		if(i1 == 0) i1 = 1;

		hkMpRational	r0(i0);
		hkMpRational	r1(i1);
		
		hkInt64	refAdd = i0 + i1;
		hkInt64	refSub = i0 - i1;
		hkInt64	refMul = i0 * i1;
		int	refDiv = i0 / i1;
		int refCmp = i0 < i1 ? -1 : (i0 > i1 ? +1 : 0);

		int				error = 0;
		hkMpRational	ret;		

		#define	SET_ERROR(_index_) { error = _index_; HK_TRACE("r0: " << r0); HK_TRACE("r1: " << r1); HK_TRACE("ret: " << ret); }
		
		// Operators +-*/

		ret = r0 + r1;
		if(hkMpRational(refAdd).compareTo(ret) != 0) SET_ERROR(1);
		
		ret = r0 - r1;
		if(hkMpRational(refSub).compareTo(ret) != 0) SET_ERROR(2);
		
		ret = r0 * r1;
		if(hkMpRational(refMul).compareTo(ret) != 0) SET_ERROR(3);
		
		ret = r0 / r1; round(ret);
		if(hkMpRational(refDiv).compareTo(ret) != 0) SET_ERROR(4);
		
		// Comparison.
		
		if(r0.compareTo(r1) != refCmp) SET_ERROR(5);

		// Bit shifts.
		{
			// x << n == x * 2^n
			const int	n = (prng.nextUint32() >> 1) % 256;
			hkMpUint	x; set(x, hkUint64(prng.nextUint32()) * ((i&1) ? 0 : hkUint64(prng.nextUint32())));
			hkMpUint	y = x; shift(y, n);
			hkMpUint	p; set(p,2); pow(p,n);
			hkMpUint	z; mul(x,p,z);
			if(compare(y,z) != 0) SET_ERROR(6);
		}

		#undef SET_ERROR
		
		if(error)
		{
			HK_TRACE("#" << i);
			HK_TRACE("Error = " << error);
			HK_TRACE("int0 = " << i0);
			HK_TRACE("int1 = " << i1);
			HK_TRACE("rat0 = " << r0);
			HK_TRACE("rat1 = " << r1);
			HK_TRACE("Ret = " << ret);
			return 0;
		}
	}
	return 1;
}

//
template <typename TU, typename TF>
static int	unitTestMpRationalConvertions(int iterations)
{
	const int					expBits = sizeof(TU) == 4 ? 8 : 11;
	hkGeometryProcessing::Prng	prng;
	for(int i=0; i<iterations; ++i)
	{
		// Generate a sample.
		TU refValue = 0;
		for(int j=0; j<(sizeof(TU) / 4); ++j)
		{
			refValue <<= 16;
			refValue <<= 16;
			refValue |= prng.nextUint32();
		}
		
		// Skip denormals.
		const TU exp = (refValue << 1) >> ((sizeof(TU) * 8) - expBits);
		if(exp == 0) continue;

		// Skip NAN.
		TF refValueF = *(TF*)&refValue;
		if(refValueF != refValueF) continue;

		// Convert back and forth.
		hkMpRational	rat; set(rat, *(TF*)&refValue);
		TU				value; get(rat, *(TF*)&value);

		// Check.
		if(hkString::memCmp(&refValue, &value, sizeof(TU)))
		{
			HK_TRACE("Iteration #"<<i);
			HK_TRACE("Ref: " << *(TF*)&refValue);
			HK_TRACE("Val: " << *(TF*)&value);
			HK_TRACE("Rat: " << rat);
			return 0;
		}
	}
	return 1;
}

//
bool	hkMpUnitTests()
{
	const int iterations = 100000;

	HK_TRACE("Basics.");
	if(!unitTestMpRationalBasics(iterations)) return false;

	HK_TRACE("Convertions floats.");
	if(!unitTestMpRationalConvertions<hkUint32,float>(iterations)) return false;

	HK_TRACE("Convertions doubles.");
	if(!unitTestMpRationalConvertions<hkUint64,double>(iterations)) return false;

	return true;
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
