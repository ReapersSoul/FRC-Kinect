//map
kernel float _map(float value, float istart, float istop, float ostart, float ostop)
{
	return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
}

//lerp
kernel float _lerp(float a, float b, float t)
{
	return a + t * (b - a);
}

struct Color{
	float r;
	float g;
	float b;
};


kernel Color ColorLerp(Color a, Color b, float t)
{
	Color c;
	c.r = _lerp(a.r, b.r, t);
	c.g = _lerp(a.g, b.g, t);
	c.b = _lerp(a.b, b.b, t);
	return c;
}

kernel Color Gradient(float t, Color* colors, int ColorCount)
{
	if (ColorCount == 0)
	{
		return Color();
	}
	if (ColorCount == 1)
	{
		return colors[0];
	}
	float t2 = _map(t, 0, 1, 0, ColorCount - 1);
	int i = (int)t2;
	t2 -= i;
	return ColorLerp(colors[i], colors[i + 1], t2);
}

kernel void DeapthToColor(float * deapthData, int deapthDataSize, Color* colors, int ColorCount, float colorClipDistanceFront_off, float colorClipDistanceFront, float colorClipDistanceBack_off, float  colorClipDistanceBack, float* output)
{
	int i = get_global_id(0);
	if (i < deapthDataSize)
	{
		float depth = deapthData[i] / 2048.0;
		float depthClipNorm = _map(depth, 0, 1, colorClipDistanceFront_off - colorClipDistanceFront, colorClipDistanceBack_off - colorClipDistanceBack);
		Color color = Gradient(depthClipNorm, colors, ColorCount);
		output[i * 3] = color.r;
		output[i * 3 + 1] = color.g;
		output[i * 3 + 2] = color.b;
	}
}