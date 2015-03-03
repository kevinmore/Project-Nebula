/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>

hkColor::Argb hkColor::s_colorTable[32] = 
{
	0xffe0e0e0, // grey
	0xffff0000,	// red
	0xff00ff00, // green
	0xff0000ff, // blue
	0xffffff00, // yellow
	0xff00ffff, // cyan
	0xffff00ff, // magenta
	0xffffffff, // white

	0xffa0a0a0, // grey
	0xffc00000,	// red
	0xff00c000, // green
	0xff0000c0, // blue
	0xffc0c000, // yellow
	0xff00c0c0, // cyan
	0xffc000c0, // magenta
	0xffc0c0c0, // white

	0xff606060, // grey
	0xff800000,	// red
	0xff008000, // green
	0xff000080, // blue
	0xff808000, // yellow
	0xff008080, // cyan
	0xff800080, // magenta
	0xff808080, // white

	0xff202020, // grey
	0xff400000,	// red
	0xff004000, // green
	0xff000040, // blue
	0xff404000, // yellow
	0xff004040, // cyan
	0xff400040, // magenta
	0xff404040  // white
};

const hkColor::Argb hkColor::MAROON = 0xFF800000;
const hkColor::Argb hkColor::DARKRED = 0xFF8B0000;
const hkColor::Argb hkColor::RED = 0xFFFF0000;
const hkColor::Argb hkColor::LIGHTPINK = 0xFFFFB6C1;
const hkColor::Argb hkColor::CRIMSON = 0xFFDC143C;
const hkColor::Argb hkColor::PALEVIOLETRED = 0xFFDB7093;
const hkColor::Argb hkColor::HOTPINK = 0xFFFF69B4;
const hkColor::Argb hkColor::DEEPPINK = 0xFFFF1493;
const hkColor::Argb hkColor::MEDIUMVIOLETRED = 0xFFC71585;
const hkColor::Argb hkColor::PURPLE = 0xFF800080;
const hkColor::Argb hkColor::DARKMAGENTA = 0xFF8B008B;
const hkColor::Argb hkColor::ORCHID = 0xFFDA70D6;
const hkColor::Argb hkColor::THISTLE = 0xFFD8BFD8;
const hkColor::Argb hkColor::PLUM = 0xFFDDA0DD;
const hkColor::Argb hkColor::VIOLET = 0xFFEE82EE;
const hkColor::Argb hkColor::FUCHSIA = 0xFFFF00FF;
const hkColor::Argb hkColor::MAGENTA = 0xFFFF00FF;
const hkColor::Argb hkColor::MEDIUMORCHID = 0xFFBA55D3;
const hkColor::Argb hkColor::DARKVIOLET = 0xFF9400D3;
const hkColor::Argb hkColor::DARKORCHID = 0xFF9932CC;
const hkColor::Argb hkColor::BLUEVIOLET = 0xFF8A2BE2;
const hkColor::Argb hkColor::INDIGO = 0xFF4B0082;
const hkColor::Argb hkColor::MEDIUMPURPLE = 0xFF9370DB;
const hkColor::Argb hkColor::SLATEBLUE = 0xFF6A5ACD;
const hkColor::Argb hkColor::MEDIUMSLATEBLUE = 0xFF7B68EE;
const hkColor::Argb hkColor::DARKBLUE = 0xFF00008B;
const hkColor::Argb hkColor::MEDIUMBLUE = 0xFF0000CD;
const hkColor::Argb hkColor::BLUE = 0xFF0000FF;
const hkColor::Argb hkColor::NAVY = 0xFF000080;
const hkColor::Argb hkColor::MIDNIGHTBLUE = 0xFF191970;
const hkColor::Argb hkColor::DARKSLATEBLUE = 0xFF483D8B;
const hkColor::Argb hkColor::ROYALBLUE = 0xFF4169E1;
const hkColor::Argb hkColor::CORNFLOWERBLUE = 0xFF6495ED;
const hkColor::Argb hkColor::LIGHTSTEELBLUE = 0xFFB0C4DE;
const hkColor::Argb hkColor::ALICEBLUE = 0xFFF0F8FF;
const hkColor::Argb hkColor::GHOSTWHITE = 0xFFF8F8FF;
const hkColor::Argb hkColor::LAVENDER = 0xFFE6E6FA;
const hkColor::Argb hkColor::DODGERBLUE = 0xFF1E90FF;
const hkColor::Argb hkColor::STEELBLUE = 0xFF4682B4;
const hkColor::Argb hkColor::DEEPSKYBLUE = 0xFF00BFFF;
const hkColor::Argb hkColor::SLATEGRAY = 0xFF708090;
const hkColor::Argb hkColor::LIGHTSLATEGRAY = 0xFF778899;
const hkColor::Argb hkColor::LIGHTSKYBLUE = 0xFF87CEFA;
const hkColor::Argb hkColor::SKYBLUE = 0xFF87CEEB;
const hkColor::Argb hkColor::LIGHTBLUE = 0xFFADD8E6;
const hkColor::Argb hkColor::TEAL = 0xFF008080;
const hkColor::Argb hkColor::DARKCYAN = 0xFF008B8B;
const hkColor::Argb hkColor::DARKTURQUOISE = 0xFF00CED1;
const hkColor::Argb hkColor::CYAN = 0xFF00FFFF;
const hkColor::Argb hkColor::MEDIUMTURQUOISE = 0xFF48D1CC;
const hkColor::Argb hkColor::CADETBLUE = 0xFF5F9EA0;
const hkColor::Argb hkColor::PALETURQUOISE = 0xFFAFEEEE;
const hkColor::Argb hkColor::LIGHTCYAN = 0xFFE0FFFF;
const hkColor::Argb hkColor::AZURE = 0xFFF0FFFF;
const hkColor::Argb hkColor::LIGHTSEAGREEN = 0xFF20B2AA;
const hkColor::Argb hkColor::TURQUOISE = 0xFF40E0D0;
const hkColor::Argb hkColor::POWDERBLUE = 0xFFB0E0E6;
const hkColor::Argb hkColor::DARKSLATEGRAY = 0xFF2F4F4F;
const hkColor::Argb hkColor::AQUAMARINE = 0xFF7FFFD4;
const hkColor::Argb hkColor::MEDIUMSPRINGGREEN = 0xFF00FA9A;
const hkColor::Argb hkColor::MEDIUMAQUAMARINE = 0xFF66CDAA;
const hkColor::Argb hkColor::SPRINGGREEN = 0xFF00FF7F;
const hkColor::Argb hkColor::MEDIUMSEAGREEN = 0xFF3CB371;
const hkColor::Argb hkColor::SEAGREEN = 0xFF2E8B57;
const hkColor::Argb hkColor::LIMEGREEN = 0xFF32CD32;
const hkColor::Argb hkColor::DARKGREEN = 0xFF006400;
const hkColor::Argb hkColor::GREEN = 0xFF008000;
const hkColor::Argb hkColor::LIME = 0xFF00FF00;
const hkColor::Argb hkColor::FORESTGREEN = 0xFF228B22;
const hkColor::Argb hkColor::DARKSEAGREEN = 0xFF8FBC8F;
const hkColor::Argb hkColor::LIGHTGREEN = 0xFF90EE90;
const hkColor::Argb hkColor::PALEGREEN = 0xFF98FB98;
const hkColor::Argb hkColor::MINTCREAM = 0xFFF5FFFA;
const hkColor::Argb hkColor::HONEYDEW = 0xFFF0FFF0;
const hkColor::Argb hkColor::CHARTREUSE = 0xFF7FFF00;
const hkColor::Argb hkColor::LAWNGREEN = 0xFF7CFC00;
const hkColor::Argb hkColor::OLIVEDRAB = 0xFF6B8E23;
const hkColor::Argb hkColor::DARKOLIVEGREEN = 0xFF556B2F;
const hkColor::Argb hkColor::YELLOWGREEN = 0xFF9ACD32;
const hkColor::Argb hkColor::GREENYELLOW = 0xFFADFF2F;
const hkColor::Argb hkColor::BEIGE = 0xFFF5F5DC;
const hkColor::Argb hkColor::LINEN = 0xFFFAF0E6;
const hkColor::Argb hkColor::LIGHTGOLDENRODYELLOW = 0xFFFAFAD2;
const hkColor::Argb hkColor::OLIVE = 0xFF808000;
const hkColor::Argb hkColor::YELLOW = 0xFFFFFF00;
const hkColor::Argb hkColor::LIGHTYELLOW = 0xFFFFFFE0;
const hkColor::Argb hkColor::IVORY = 0xFFFFFFF0;
const hkColor::Argb hkColor::DARKKHAKI = 0xFFBDB76B;
const hkColor::Argb hkColor::KHAKI = 0xFFF0E68C;
const hkColor::Argb hkColor::PALEGOLDENROD = 0xFFEEE8AA;
const hkColor::Argb hkColor::WHEAT = 0xFFF5DEB3;
const hkColor::Argb hkColor::GOLD = 0xFFFFD700;
const hkColor::Argb hkColor::LEMONCHIFFON = 0xFFFFFACD;
const hkColor::Argb hkColor::PAPAYAWHIP = 0xFFFFEFD5;
const hkColor::Argb hkColor::DARKGOLDENROD = 0xFFB8860B;
const hkColor::Argb hkColor::GOLDENROD = 0xFFDAA520;
const hkColor::Argb hkColor::ANTIQUEWHITE = 0xFFFAEBD7;
const hkColor::Argb hkColor::CORNSILK = 0xFFFFF8DC;
const hkColor::Argb hkColor::OLDLACE = 0xFFFDF5E6;
const hkColor::Argb hkColor::MOCCASIN = 0xFFFFE4B5;
const hkColor::Argb hkColor::NAVAJOWHITE = 0xFFFFDEAD;
const hkColor::Argb hkColor::ORANGE = 0xFFFFA500;
const hkColor::Argb hkColor::BISQUE = 0xFFFFE4C4;
const hkColor::Argb hkColor::TAN = 0xFFD2B48C;
const hkColor::Argb hkColor::DARKORANGE = 0xFFFF8C00;
const hkColor::Argb hkColor::BURLYWOOD = 0xFFDEB887;
const hkColor::Argb hkColor::SADDLEBROWN = 0xFF8B4513;
const hkColor::Argb hkColor::SANDYBROWN = 0xFFF4A460;
const hkColor::Argb hkColor::BLANCHEDALMOND = 0xFFFFEBCD;
const hkColor::Argb hkColor::LAVENDERBLUSH = 0xFFFFF0F5;
const hkColor::Argb hkColor::SEASHELL = 0xFFFFF5EE;
const hkColor::Argb hkColor::FLORALWHITE = 0xFFFFFAF0;
const hkColor::Argb hkColor::SNOW = 0xFFFFFAFA;
const hkColor::Argb hkColor::PERU = 0xFFCD853F;
const hkColor::Argb hkColor::PEACHPUFF = 0xFFFFDAB9;
const hkColor::Argb hkColor::CHOCOLATE = 0xFFD2691E;
const hkColor::Argb hkColor::SIENNA = 0xFFA0522D;
const hkColor::Argb hkColor::LIGHTSALMON = 0xFFFFA07A;
const hkColor::Argb hkColor::CORAL = 0xFFFF7F50;
const hkColor::Argb hkColor::DARKSALMON = 0xFFE9967A;
const hkColor::Argb hkColor::MISTYROSE = 0xFFFFE4E1;
const hkColor::Argb hkColor::ORANGERED = 0xFFFF4500;
const hkColor::Argb hkColor::SALMON = 0xFFFA8072;
const hkColor::Argb hkColor::TOMATO = 0xFFFF6347;
const hkColor::Argb hkColor::ROSYBROWN = 0xFFBC8F8F;
const hkColor::Argb hkColor::PINK = 0xFFFFC0CB;
const hkColor::Argb hkColor::INDIANRED = 0xFFCD5C5C;
const hkColor::Argb hkColor::LIGHTCORAL = 0xFFF08080;
const hkColor::Argb hkColor::BROWN = 0xFFA52A2A;
const hkColor::Argb hkColor::FIREBRICK = 0xFFB22222;
const hkColor::Argb hkColor::BLACK = 0xFF000000;
const hkColor::Argb hkColor::DIMGRAY = 0xFF696969;
const hkColor::Argb hkColor::GRAY = 0xFF808080;
const hkColor::Argb hkColor::DARKGRAY = 0xFFA9A9A9;
const hkColor::Argb hkColor::SILVER = 0xFFC0C0C0;
const hkColor::Argb hkColor::LIGHTGREY = 0xFFD3D3D3;
const hkColor::Argb hkColor::GAINSBORO = 0xFFDCDCDC;
const hkColor::Argb hkColor::WHITESMOKE = 0xFFF5F5F5;
const hkColor::Argb hkColor::WHITE = 0xFFFFFFFF;
const hkColor::Argb hkColor::GREY		= 0xff888888;
const hkColor::Argb hkColor::GREY25 = 0xff404040;
const hkColor::Argb hkColor::GREY50 = 0xff808080;
const hkColor::Argb hkColor::GREY75 = 0xffc0c0c0;

// Havok product colors
const hkColor::Argb hkColor::PHYSICS = 0xFFFFB300;
const hkColor::Argb hkColor::DESTRUCTION = 0xFFDB0020;
const hkColor::Argb hkColor::ANIMATION = 0xFF02A22B;
const hkColor::Argb hkColor::BEHAVIOR = 0xFF3370B8;
const hkColor::Argb hkColor::CLOTH = 0xFFB29CDC;
const hkColor::Argb hkColor::AI = 0xFFACCEF0;
const hkColor::Argb hkColor::SCRIPT = 0xFFBFB630;

hkColor::Argb HK_CALL hkColor::rgbFromChars(unsigned char red, unsigned char green, unsigned char blue, unsigned char alpha)
{
	hkColor::Argb color =  (static_cast<hkColor::Argb>(alpha) * (256 * 256 * 256)) 
					+ (static_cast<hkColor::Argb>(red) * (256 * 256))
					+ (static_cast<hkColor::Argb>(green) * 256)
					+ static_cast<hkColor::Argb>(blue);
	return color;
}

hkColor::Argb HK_CALL hkColor::rgbFromFloats(const hkReal red, const hkReal green, const hkReal blue, const hkReal alpha)
{
	HK_ASSERT2(0x5d228af1,  (alpha >= 0.0f) && (alpha <= 1.0f), "Color component is out of range!" );
	HK_ASSERT2(0x293c6824,  (red >= 0.0f) && (red <= 1.0f), "Color component is out of range!" );
	HK_ASSERT2(0x3df5f1a3,  (green >= 0.0f) && (green <= 1.0f), "Color component is out of range!" );
	HK_ASSERT2(0x293ff0fe,  (blue >= 0.0f) && (blue <= 1.0f), "Color component is out of range!" );

	unsigned char cAlpha = (unsigned char)(hkMath::hkFloatToInt(alpha * 255));
	unsigned char cRed   = (unsigned char)(hkMath::hkFloatToInt(red * 255));
	unsigned char cGreen = (unsigned char)(hkMath::hkFloatToInt(green * 255));
	unsigned char cBlue	 = (unsigned char)(hkMath::hkFloatToInt(blue * 255));

	return rgbFromChars(cRed, cGreen, cBlue, cAlpha);
}


// Convert from HSV color values to RGB values (all in range [0,1])
static inline void _HSVtoRGB( hkReal *r, hkReal *g, hkReal *b, hkReal h, hkReal s, hkReal v )
{
	int i; hkReal f, p, q, t;
	if ( s==0.0f ) 
	{ 
		*r = *g = *b = v; 
		return; 
	}
	if ( h==1.0f ) 
	{ 
		i=5; 
		h=6.0f; 
	}
	else 
	{ 
		h *= 6.0f;  
		i = static_cast<int>( hkMath::floor(h) ); 
	}

	f = h - i; 
	p = v * (1.0f - s);
	q = v * (1.0f - s*f);  
	t = v * (1.0f - s*(1.0f-f));
	switch(i) 
	{
		case 0:  *r = v; *g = t; *b = p; break;
		case 1:  *r = q; *g = v; *b = p; break;
		case 2:  *r = p; *g = v; *b = t; break;
		case 3:  *r = p; *g = q; *b = v; break;
		case 4:  *r = t; *g = p; *b = v; break;
		default: *r = v; *g = p; *b = q; break;
	}
}


hkColor::Argb HK_CALL hkColor::rgbFromHSV(const hkReal h, const hkReal s, const hkReal v, const hkReal alpha)
{
	HK_ASSERT2(0x5d228af2,  (alpha >= 0.0f) && (alpha <= 1.0f), "Color component is out of range!" );
	HK_ASSERT2(0x293c6825,  (h >= 0.0f) && (h <= 1.0f), "Color component is out of range!" );
	HK_ASSERT2(0x3df5f1a7,  (s >= 0.0f) && (s <= 1.0f), "Color component is out of range!" );
	HK_ASSERT2(0x293ff0fb,  (v >= 0.0f) && (v <= 1.0f), "Color component is out of range!" );

	hkReal r, g, b;
	_HSVtoRGB(&r, &g, &b, h, s, v);

	return rgbFromFloats(r, g, b, alpha);
}


hkColor::Argb HK_CALL hkColor::getRandomColor()
{
	static hkPseudoRandomGenerator prng(0);
	return getRandomColor(prng);
}

hkColor::Argb HK_CALL hkColor::getRandomColor( hkPseudoRandomGenerator& rand)
{
	hkReal r = rand.getRandRange(0.0f, 1.0f);
	hkReal g = rand.getRandRange(0.0f, 1.0f);
	hkReal b = rand.getRandRange(0.0f, 1.0f);
	hkReal a = 1.0f;

	return rgbFromFloats(r,g,b,a);
}

hkColor::Argb HK_CALL hkColor::getSpectrumColor(hkReal value)
{
	hkVector4	palette[6];
	palette[0].set(0,0,0,1);
	palette[1].set(0,0,1,1);
	palette[2].set(0,1,1,1);
	palette[3].set(0,1,0,1);
	palette[4].set(1,1,0,1);
	palette[5].set(1,0,0,1);
	hkVector4		color; color.setZero();
	if(value <= 0.0f)		color = palette[0];
	else if(value >= 1.0f)	color = palette[5];
	else
	{
		hkReal		ms = value * 5.0f;
		int			c = (int) ms;
		hkSimdReal	u; u.setFromFloat(ms - c);
		color.setInterpolate(palette[c], palette[c+1], u);
	}
	return hkColor::rgbFromFloats(color(0),color(1),color(2),1);
}

hkColor::Argb HK_CALL hkColor::getPaletteColor(int i, unsigned char alpha)
{
	return (s_colorTable[i % HK_COUNT_OF(s_colorTable)] & 0x00ffffffU) | (static_cast<hkColor::Argb>( alpha )<<24);
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
