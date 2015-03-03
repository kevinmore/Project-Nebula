/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Vector/hkVector2f.h>

void hkVector2f::setClosestPointOnSegmentToPoint( const hkVector2f& a, const hkVector2f& b, const hkVector2f& p)
{
	hkVector2f edge; edge.setSub(a, b);
	hkVector2f toa; toa.setSub(a,p);
	hkVector2f tob; tob.setSub(b,p);
	hkFloat32 dota = edge.dot(toa);
	hkFloat32 dotb = edge.dot(tob);
	if( dota * dotb < 0 )           
	{
		this->setInterpolate( a, b, dota/(dota-dotb) );
	}
	else if( dota < 0 )
	{
		*this = a;
	}
	else
	{
		*this = b;
	}
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
