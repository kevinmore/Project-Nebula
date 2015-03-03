/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

void hkVector4d::setRotatedDir(const hkMatrix3d& a, hkVector4dParameter b)
{
	_setRotatedDir(a,b);
}

void hkVector4d::setRotatedInverseDir(const hkMatrix3d& a, hkVector4dParameter b)
{
	_setRotatedInverseDir(a,b);
}

void hkVector4d::setTransformedPos(const hkTransformd& a, hkVector4dParameter b)
{
	_setTransformedPos(a,b);
}

void hkVector4d::setTransformedInversePos(const hkTransformd& a, hkVector4dParameter b)
{
	_setTransformedInversePos(a,b);
}

void hkVector4d::setTransformedPos(const hkQsTransformd& a, hkVector4dParameter b)
{
	_setTransformedPos(a,b);
}

void hkVector4d::setTransformedInversePos(const hkQsTransformd& a, hkVector4dParameter b)
{
	_setTransformedInversePos(a,b);
}

void hkVector4d::setTransformedPos(const hkQTransformd& a, hkVector4dParameter b)
{
	_setTransformedPos(a,b);
}

void hkVector4d::setTransformedInversePos(const hkQTransformd& a, hkVector4dParameter b)
{
	_setTransformedInversePos(a,b);
}

void hkVector4d::setRotatedDir(hkQuaterniondParameter quat, hkVector4dParameter direction)
{
	_setRotatedDir(quat,direction);
}

void hkVector4d::setRotatedInverseDir(hkQuaterniondParameter quat, hkVector4dParameter direction)
{
	_setRotatedInverseDir(quat,direction);
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
